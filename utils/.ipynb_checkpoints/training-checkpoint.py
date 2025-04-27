# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from tqdm import tqdm
from copy import deepcopy
import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel

from utils import random_id
from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import *
from utils.status import ProgressBar

from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

try:
    import wandb
except ImportError:
    wandb = None

def imbalanced_trainloader_transfer(train_loader: DataLoader, t: int, task: int, num1: int, num2: int, balanced_sampling: bool = False) -> DataLoader:
    """
    Creates an imbalanced dataset scenario based on the current task, with an optional balanced sampling across labels.
    
    Args:
        train_loader: Original train_loader to sample from.
        t: The current task index.
        task: The target task index for imbalanced sampling.
        num1: Number of samples to select if t equals the task.
        num2: Number of samples to select if t does not equal the task.
        balanced_sampling: If True, performs balanced sampling across labels.

    Returns:
        A new DataLoader with the selected number of samples.
    """
    # Extract all data and labels from the train_loader
    all_inputs, all_labels, all_not_aug_inputs, all_logits = [], [], [], []
    for batch in train_loader:
        if len(batch) == 4:  # If logits are present
            inputs, labels, not_aug_inputs, logits = batch
            all_inputs.append(inputs)
            all_labels.append(labels.unsqueeze(0) if labels.dim() == 0 else labels)
            all_not_aug_inputs.append(not_aug_inputs)
            all_logits.append(logits)
        else:  # If logits are not present
            inputs, labels, not_aug_inputs = batch
            all_inputs.append(inputs)
            all_labels.append(labels.unsqueeze(0) if labels.dim() == 0 else labels)
            all_not_aug_inputs.append(not_aug_inputs)

    # Concatenate all tensors
    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)
    all_not_aug_inputs = torch.cat(all_not_aug_inputs)
    all_logits = torch.cat(all_logits) if all_logits else None

    # Determine the number of samples to select
    num_samples = num1 if t == task else num2
    num_samples = min(num_samples, all_inputs.size(0))  # Ensure num_samples is within the valid range

    if balanced_sampling:
        # Perform balanced sampling across labels
        unique_labels = all_labels.unique()
        samples_per_label = num_samples // len(unique_labels)
        remaining_samples = num_samples % len(unique_labels)
        selected_indices = []

        for label in unique_labels:
            label_indices = (all_labels == label).nonzero(as_tuple=True)[0]
            selected_indices.extend(label_indices[torch.randperm(len(label_indices))[:samples_per_label]].tolist())
        
        # Randomly allocate remaining samples across different labels
        if remaining_samples > 0:
            remaining_indices = torch.cat([label_indices for label in unique_labels])
            selected_indices.extend(remaining_indices[torch.randperm(len(remaining_indices))[:remaining_samples]].tolist())
        
        selected_indices = torch.tensor(selected_indices)
    else:
        # Randomly select indices
        selected_indices = torch.randperm(all_inputs.size(0))[:num_samples]

    # Create the subset of data based on selected indices
    subset_inputs = all_inputs[selected_indices]
    subset_labels = all_labels[selected_indices]
    subset_not_aug_inputs = all_not_aug_inputs[selected_indices]
    
    if all_logits is not None:
        subset_logits = all_logits[selected_indices]
        dataset = TensorDataset(subset_inputs, subset_labels, subset_not_aug_inputs, subset_logits)
    else:
        dataset = TensorDataset(subset_inputs, subset_labels, subset_not_aug_inputs)
    
    # Create a new DataLoader with the selected subset
    new_train_loader = DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=True)

    # Count and print the label distribution in the new train_loader
    label_count = Counter(subset_labels.tolist())
    print("Label distribution in the new train_loader:", label_count)

    return new_train_loader

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs, _ = model(inputs, k)
                # outputs = model(inputs, k)
            else:
                outputs, _ = model(inputs)
                # outputs = model(inputs)
            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = random_id(5)
    name = f'{run_name}_{run_id}'
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name)
    args.wandb_url = wandb.run.get_url()


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, _ = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            model.meta_end_task(dataset)

    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)

        if not args.disable_log and past_res is not None:
            (results, results_mask_classes, csvdump) = past_res
            logger.load(csvdump)

        print('Checkpoint Loaded!')

    progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)

    if args.enable_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    start_task = 0 if args.start_from is None else args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

    torch.cuda.empty_cache()
    for t in range(start_task, end_task):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        ############################################################
        # train_loader = imbalanced_trainloader_transfer(train_loader, t, task=model.imb_task, num1=4000, num2=400, balanced_sampling=True)
        ############################################################
        model.meta_begin_task(dataset)

        if not args.inference_only:
            if t and args.enable_other_metrics:
                accs = evaluate(model, dataset, last=True)
                results[t - 1] = results[t - 1] + accs[0]
                if dataset.SETTING == 'class-il':
                    results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

            scheduler = dataset.get_scheduler(model, args) if not hasattr(model, 'scheduler') else model.scheduler
            # 循环中间做sample selection，一个task结束后在做sample，做off-online的对比；
            for epoch in range(model.args.n_epochs):
                train_iter = iter(train_loader)
                data_len = None
                if not isinstance(dataset, GCLDataset):
                    data_len = len(train_loader)
                i = 0
                while True:
                    # 样本循环
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break
                    if args.debug_mode and i > model.get_debug_iters():
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                    else:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
                    assert not math.isnan(loss)
                    progress_bar.prog(i, data_len, epoch, t, loss)
                    i += 1

                if scheduler is not None:
                    scheduler.step()

                if args.eval_epochs is not None and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs - 1:
                    epoch_accs = evaluate(model, dataset)

                    log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)

        model.meta_end_task(dataset)
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        log_accs(args, logger, accs, t, dataset.SETTING)

        if args.savecheck:
            save_obj = {
                'model': model.state_dict(),
                'args': args,
                'results': [results, results_mask_classes, logger.dump()],
                'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            if 'buffer_size' in model.args:
                save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

            # Saving model checkpoint
            checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
            torch.save(save_obj, checkpoint_name)
            
        info_sample_data, info_sample_labels, info_sample_logits = model.get_info_sample(train_loader)
        model.main_buffer.add_data(info_sample_data, info_sample_labels, info_sample_logits, model=None)
        model.reset_temp_buffer()
        model.print_buffer_info()
        
    
    if args.validation:
        del dataset
        args.validation = None

        final_dataset = get_dataset(args)
        for _ in range(final_dataset.N_TASKS):
            final_dataset.get_data_loaders()
        accs = evaluate(model, final_dataset)
        log_accs(args, logger, accs, t, final_dataset.SETTING, prefix="FINAL")

    if not args.disable_log and args.enable_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()

##################################################################################################
# def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
#     """
#     The training process, including evaluations and loggers.

#     Args:
#         model: the module to be trained
#         dataset: the continual dataset at hand
#         args: the arguments of the current execution
#     """
#     print(args)
#     # sys.stdout.flush()  # 确保立即打印

#     if not args.nowand:
#         initialize_wandb(args)

#     model.net.to(model.device)
#     results, results_mask_classes = [], []

#     if not args.disable_log:
#         logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

#     if args.start_from is not None:
#         for i in range(args.start_from):
#             train_loader, _ = dataset.get_data_loaders()
#             model.meta_begin_task(dataset)
#             model.meta_end_task(dataset)

#     if args.loadcheck is not None:
#         model, past_res = mammoth_load_checkpoint(args, model)

#         if not args.disable_log and past_res is not None:
#             (results, results_mask_classes, csvdump) = past_res
#             logger.load(csvdump)

#         print('Checkpoint Loaded!')
#         # sys.stdout.flush()  # 确保立即打印

#     progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)

#     if args.enable_other_metrics:
#         dataset_copy = get_dataset(args)
#         for t in range(dataset.N_TASKS):
#             model.net.train()
#             _, _ = dataset_copy.get_data_loaders()
#         if model.NAME != 'icarl' and model.NAME != 'pnn':
#             random_results_class, random_results_task = evaluate(model, dataset_copy)

#     print(file=sys.stderr)
#     # sys.stdout.flush()  # 确保立即打印

#     start_task = 0 if args.start_from is None else args.start_from
#     end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

#     torch.cuda.empty_cache()
#     for t in range(start_task, end_task):
#         model.net.train()
#         train_loader, test_loader = dataset.get_data_loaders()
#         model.meta_begin_task(dataset)

#         if not args.inference_only:
#             if t and args.enable_other_metrics:
#                 accs = evaluate(model, dataset, last=True)
#                 results[t - 1] = results[t - 1] + accs[0]
#                 if dataset.SETTING == 'class-il':
#                     results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

#             scheduler = dataset.get_scheduler(model, args) if not hasattr(model, 'scheduler') else model.scheduler

#             for epoch in range(model.args.n_epochs):
#                 train_iter = iter(train_loader)
#                 data_len = None
#                 if not isinstance(dataset, GCLDataset):
#                     data_len = len(train_loader)
#                 i = 0
#                 while True:
#                     try:
#                         data = next(train_iter)
#                     except StopIteration:
#                         break
#                     if args.debug_mode and i > model.get_debug_iters():
#                         break
#                     if hasattr(dataset.train_loader.dataset, 'logits'):
#                         inputs, labels, not_aug_inputs, logits = data
#                         inputs = inputs.to(model.device)
#                         labels = labels.to(model.device, dtype=torch.long)
#                         not_aug_inputs = not_aug_inputs.to(model.device)
#                         logits = logits.to(model.device)
#                         loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
#                     else:
#                         inputs, labels, not_aug_inputs = data
#                         inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
#                         not_aug_inputs = not_aug_inputs.to(model.device)
#                         loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
#                     assert not math.isnan(loss)
#                     progress_bar.prog(i, data_len, epoch, t, loss)
#                     i += 1

#                 if scheduler is not None:
#                     scheduler.step()

#                 if args.eval_epochs is not None and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs - 1:
#                     epoch_accs = evaluate(model, dataset)

#                     log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)
# #############################################################################################################################
#             # info_sample_data, info_sample_labels, info_sample_logits = model.get_info_sample(train_loader)
#             # model.main_buffer.add_data(info_sample_data, info_sample_labels, info_sample_logits, model=None)
#             # model.reset_temp_buffer()
# #############################################################################################################################
#         model.meta_end_task(dataset)
#         accs = evaluate(model, dataset)
#         results.append(accs[0])
#         results_mask_classes.append(accs[1])
#         log_accs(args, logger, accs, t, dataset.SETTING)
#         # sys.stdout.flush()  # 确保立即打印
        
#         info_sample_data, info_sample_labels, info_sample_logits = model.get_info_sample(train_loader)
#         # model.main_buffer.add_data(info_sample_data, info_sample_labels, info_sample_logits, model=None)
        
#         if args.savecheck:
#             save_obj = {
#                 'model': model.state_dict(),
#                 'args': args,
#                 'results': [results, results_mask_classes, logger.dump()],
#                 'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
#                 'scheduler': scheduler.state_dict() if scheduler is not None else None,
#             }
#             if 'buffer_size' in model.args:
#                 save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

#             checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
#             torch.save(save_obj, checkpoint_name)

#     if args.validation:
#         del dataset
#         args.validation = None

#         final_dataset = get_dataset(args)
#         for _ in range(final_dataset.N_TASKS):
#             final_dataset.get_data_loaders()
#         accs = evaluate(model, final_dataset)
#         log_accs(args, logger, accs, t, final_dataset.SETTING, prefix="FINAL")

#     if not args.disable_log and args.enable_other_metrics:
#         logger.add_bwt(results, results_mask_classes)
#         logger.add_forgetting(results, results_mask_classes)
#         if model.NAME != 'icarl' and model.NAME != 'pnn':
#             logger.add_fwt(results, random_results_class, results_mask_classes, random_results_task)

#     if not args.disable_log:
#         logger.write(vars(args))
#         if not args.nowand:
#             d = logger.dump()
#             d['wandb_url'] = wandb.run.get_url()
#             wandb.log(d)

#     if not args.nowand:
#         wandb.finish()
#     # sys.stdout.flush()  # 确保立即打印
##################################################################################################

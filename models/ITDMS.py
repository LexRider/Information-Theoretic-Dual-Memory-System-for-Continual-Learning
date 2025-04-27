import torch
import time
from torch import nn
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.my_buffer_new import MyBuffer
from utils.buffer import Buffer
from tqdm import tqdm
from torch_optimizer import Lookahead
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import datetime

lamd_H2 = -1
lamd_CS = 1
lamd_Bernoulli = 0.5
sample_ratio = 0.5
epsilon = 1e-10
MAX_STEPS = 100000
main_buffer_size = 0.75
temp_buffer_size = 0.25

ksp_reg = 5
L1_reg = 1
entropy_reg = 1


class MyModel(ContinualModel):
    NAME = 'my_model_mt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    output_logits = None

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning with My_Model.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--replay_times', type=int, default=1, help='Number of times to replay buffer data.')

        parser.add_argument('--lamd_H2', type=float, default=-1, help='H2 regularizer of dataset.')
        parser.add_argument('--lamd_CS', type=float, default=1, help='CS regularizer of dataset.')
        parser.add_argument('--band_width', type=float, default=0.01, help='band width of KDE estimation.')
        parser.add_argument('--ksp_reg', type=float, default=5, help='ksp regularizer of sample weight.')
        parser.add_argument('--l1_reg', type=float, default=1, help='L1 regularizer of sample weight.')
        parser.add_argument('--entropy_reg', type=float, default=1, help='entropy regularizer of sample weight.')
        parser.add_argument('--weight_init', type=float, default=0.1, help='init number of sample weight.')
        parser.add_argument('--stop_sample', type=float, default=97, help='stop number of info sample process.')
        parser.add_argument('--sample_lr', type=float, default=0.01, help='learning rate of info sample process.')

        parser.add_argument('--main_batch', type=float, default=1.0, help='replay batch size from main buffer.')
        parser.add_argument('--tmp_batch', type=float, default=0.05, help='replay batch size from tmp buffer.')

        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.main_buffer = MyBuffer(int(self.args.buffer_size * self.args.main_batch))
        self.temp_buffer = Buffer(int(self.args.buffer_size * self.args.tmp_batch))  # 临时缓冲区

        # self.main_buffer = MyBuffer(self.args.buffer_size)
        # self.temp_buffer = Buffer(self.args.buffer_size)  # 临时缓冲区

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.replay_times = args.replay_times
        self.prev_loss = float('inf')
        self.stable_loss = False
        self.new_task_started = False
        self.current_task_finished = False
        self.buffer_shrink_started = False
        self.is_first_task = False
        self.total_shrink_steps = 100
        self.current_shrink_step = 0
        self.task_counter = 0
        self.counter = 0
        self.tmp_step = 1
        self.discrepancy_rate = 0
        self.amplify_ratio = 1.1
        self.H2_task_data = None

        self.lamd_H2 = args.lamd_H2
        self.lamd_CS = args.lamd_CS
        self.band_width = args.band_width
        self.ksp_reg = args.ksp_reg
        self.l1_reg = args.l1_reg
        self.entropy_reg = args.entropy_reg
        self.weight_init = args.weight_init
        self.stop_sample = args.stop_sample
        self.sample_lr = args.sample_lr
        self.main_batch = args.main_batch
        self.tmp_batch = args.tmp_batch


        print("lamd_H2:", self.lamd_H2,
              " lamd_CS:", self.lamd_CS,
              " band_width:", self.band_width,
              " sample_ratio:", self.amplify_ratio,
              " ksp_reg:", self.ksp_reg,
              " reg_reg:", self.l1_reg,
              " entropy_reg:", self.entropy_reg,
              " weight_init:", self.weight_init,
              " stop_sample:", self.stop_sample
              )

    def print_buffer_info(self):
        def get_label_info(buffer):
            if buffer is None or buffer.is_empty() or buffer.labels is None or len(buffer.labels) == 0:
                return {}
            labels, counts = buffer.labels.unique(return_counts=True)
            return dict(zip(labels.tolist(), counts.tolist()))

        main_buffer_size = len(self.main_buffer.examples) if not self.main_buffer.is_empty() else 0
        temp_buffer_size = len(self.temp_buffer.examples) if not self.temp_buffer.is_empty() else 0
        main_buffer_info = get_label_info(self.main_buffer)
        temp_buffer_info = get_label_info(self.temp_buffer)

        print("Main Buffer Label Info:", main_buffer_size, "/", main_buffer_info)
        print("Temp Buffer Label Info:", temp_buffer_size, "/", temp_buffer_info)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.counter += 1

        self.opt.zero_grad()
        outputs, _ = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        self.temp_buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.detach())

        # 从两个buffer中获取数据
        if not self.temp_buffer.is_empty():
            if self.tmp_batch != 0:
                if self.main_buffer.is_empty():
                    buf_inputs, _, buf_logits = self.temp_buffer.get_data(self.args.minibatch_size,
                                                                          transform=self.transform,
                                                                          device=self.device)
                else:
                    buf_inputs, _, buf_logits = self.main_buffer.get_data(int(self.args.minibatch_size * self.main_batch),transform=self.transform,device=self.device)
                    temp_buf_inputs, _, temp_buf_logits = self.temp_buffer.get_data(int(self.args.minibatch_size * self.tmp_batch),transform=self.transform,device=self.device)
                    buf_inputs = torch.cat((buf_inputs, temp_buf_inputs), dim=0)
                    buf_logits = torch.cat((buf_logits, temp_buf_logits), dim=0)

                buf_outputs, _ = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                if self.main_buffer.is_empty():
                    buf_inputs, buf_labels, _ = self.temp_buffer.get_data(self.args.minibatch_size,transform=self.transform,device=self.device)
                else:
                    buf_inputs, buf_labels, _ = self.main_buffer.get_data(int(self.args.minibatch_size * self.main_batch),transform=self.transform,device=self.device)
                    temp_buf_inputs, temp_buf_labels, _ = self.temp_buffer.get_data(int(self.args.minibatch_size * self.tmp_batch),transform=self.transform,device=self.device)
                    buf_inputs = torch.cat((buf_inputs, temp_buf_inputs), dim=0)
                    buf_labels = torch.cat((buf_labels, temp_buf_labels), dim=0)

                buf_outputs, _ = self.net(buf_inputs)
                loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                loss_ce.backward()
                tot_loss += loss_ce.item()

            else:
                if not self.main_buffer.is_empty():
                    buf_inputs, _, buf_logits = self.main_buffer.get_data(self.args.minibatch_size,transform=self.transform,device=self.device)
                    buf_outputs, _ = self.net(buf_inputs)
                    loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                    loss_mse.backward()
                    tot_loss += loss_mse.item()

                    buf_inputs, buf_labels, _ = self.main_buffer.get_data(self.args.minibatch_size,transform=self.transform,device=self.device)
                    buf_outputs, _ = self.net(buf_inputs)
                    loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                    loss_ce.backward()
                    tot_loss += loss_ce.item()

        self.opt.step()

        return tot_loss

    def reset_temp_buffer(self):
        self.temp_buffer = Buffer(self.main_buffer.buffer_size)

    def get_replay_batch_from_buffer(self, sample_num):
        if self.main_buffer.examples is None and self.temp_buffer.examples is None:
            return None, None, None, None, None

        selected_data = []
        selected_labels = []
        selected_logits = []

        if self.main_buffer.examples is not None:
            # 主缓冲区保持样本均衡选择
            unique_labels, label_counts = self.main_buffer.labels.unique(return_counts=True)
            samples_per_label = sample_num // len(unique_labels)

            for label in unique_labels:
                indices = (self.main_buffer.labels == label).nonzero(as_tuple=True)[0]
                selected_indices = indices[torch.randperm(len(indices))[:samples_per_label]]
                selected_data.append(self.main_buffer.examples[selected_indices])
                selected_labels.append(self.main_buffer.labels[selected_indices])
                selected_logits.append(self.main_buffer.logits[selected_indices])

        if self.temp_buffer.examples is not None:
            # 临时缓冲区随机选择样本
            indices = torch.randperm(len(self.temp_buffer.examples))[:sample_num]
            selected_data.append(self.temp_buffer.examples[indices])
            selected_labels.append(self.temp_buffer.labels[indices])
            selected_logits.append(self.temp_buffer.logits[indices])

        if selected_data:
            X = torch.cat(selected_data)
            y = torch.cat(selected_labels)
            logits = torch.cat(selected_logits)
        else:
            X, y, logits = None, None, None

        return X, y, logits, None, None

    def get_info_sample(self, data_loader, strategy="balanced"):
        data_list = []
        label_list = []
        not_aug_list = []
        for inputs, labels, not_aug_inputs in data_loader:
            data_list.append(inputs)
            label_list.append(labels)
            not_aug_list.append(not_aug_inputs)

        all_aug_data = torch.cat(data_list, dim=0).to(self.device)
        all_labels = torch.cat(label_list, dim=0).to(self.device)
        all_not_aug = torch.cat(not_aug_list, dim=0).to(self.device)
        print(f"当前任务的数据量：{all_aug_data.shape}, {all_labels.shape}, {all_not_aug.shape}")
        #######################################################################################################
        # 提取样本特征
        batch_size = 128  # 定义批处理大小
        num_batches = (len(all_not_aug) + batch_size - 1) // batch_size
        feature_data_list = []

        with torch.no_grad():
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(all_not_aug))
                batch_not_aug = all_not_aug[batch_start:batch_end]
                _, batch_feature_data = self.net(batch_not_aug)
                batch_feature_data = batch_feature_data.detach()
                feature_data_list.append(batch_feature_data)

        feature_data = torch.cat(feature_data_list, dim=0).to(self.device)
        #######################################################################################################

        if strategy == "balanced":
            print("info sample start, strategy: Balanced")
            # 标签均等采样策略
            unique_labels, label_counts = all_labels.unique(return_counts=True)
            num_samples_per_label = self.args.buffer_size // len(unique_labels)
            selected_indices = []

            for label in unique_labels:
                label_indices = (all_labels == label).nonzero(as_tuple=True)[0]

                # # 在特征空间中计算sample_weight
                # label_data = feature_data[label_indices]

                # # 调用 sample_from_dataloader 函数进行采样
                # sample_ratio_end = self.amplify_ratio * num_samples_per_label / len(label_data)
                # sample_weight = self.sample_from_dataloader(label_data, sample_ratio_end, show_info=True)
                # selected_label_indices = torch.multinomial(sample_weight, num_samples_per_label, replacement=False)

                # 在样本空间中计算sample_weight
                data = all_not_aug[label_indices]

                # 调用 sample_from_dataloader 函数进行采样
                sample_ratio_end = self.amplify_ratio * num_samples_per_label / len(data)
                sample_weight = self.sample_from_dataloader(data, sample_ratio_end, show_info=True)
                selected_label_indices = torch.multinomial(sample_weight, num_samples_per_label, replacement=False)

                selected_indices.append(label_indices[selected_label_indices])

            selected_indices = torch.cat(selected_indices)

        else:
            print("info sample start, strategy: Unbalanced")
            # 默认策略
            sample_ratio_end = self.amplify_ratio * self.args.buffer_size / len(all_not_aug)

            # # 在特征空间中计算sample_weight
            # sample_weight = self.sample_from_dataloader(feature_data.detach(), sample_ratio_end, show_info=True)

            # 在样本空间中计算sample_weight
            sample_weight = self.sample_from_dataloader(all_not_aug, sample_ratio_end, show_info=True)

            selected_indices = torch.multinomial(sample_weight, self.args.buffer_size, replacement=False)

        info_sample_data = all_not_aug[selected_indices]
        info_sample_labels = all_labels[selected_indices]

        info_sample_logits = []
        batch_size = 128  # 重新定义一个适当的批处理大小
        num_batches = (len(selected_indices) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(selected_indices))
            batch_indices = selected_indices[batch_start:batch_end]
            batch_logits, _ = self.net(all_aug_data[batch_indices])
            batch_logits = batch_logits.detach()
            info_sample_logits.append(batch_logits)

        info_sample_logits = torch.cat(info_sample_logits, dim=0)

        return info_sample_data, info_sample_labels, info_sample_logits

    def sample_from_dataloader(self, data, sample_ratio, show_info=False, max_steps=MAX_STEPS, init_method="constant"):
        device = data.device
        data_size = data.size(0)
        sample_num = int(data_size * sample_ratio)

        # 初始化样本权重，使用指定的方法
        sample_weight = self.initialize_sample_weight(data_size, method=init_method)
        # sample_weight = torch.rand(data_size, device=device, requires_grad=True)

        base_optimizer = torch.optim.NAdam(params=[sample_weight], lr=self.sample_lr)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

        for step in range(max_steps):
            # for step in tqdm(range(max_steps)):

            # if step == 0:
            #   break

            optimizer.zero_grad()

            u = torch.rand(data_size, device=device)
            temp = torch.log(sample_weight + epsilon) - torch.log(1 - sample_weight + epsilon) + torch.log(
                u + epsilon) - torch.log(1 - u + epsilon)
            temp /= lamd_Bernoulli
            p_Bernoulli = torch.sigmoid(temp)

            # # 在特征空间中计算sample_weight
            # sample_data = p_Bernoulli.view(-1, 1) * data

            # 在样本空间中计算sample_weight
            sample_data = p_Bernoulli.view(-1, 1, 1, 1) * data

            info_cost, H2, CS = self.get_info_cost(data, sample_data, minibatch=False)

            # 计算kspaser正则项
            l1_regularization_kspaser = torch.abs(sample_num - torch.norm(sample_weight, p=1))
            # 计算L1正则项
            l1_regularization = torch.norm(sample_weight, p=1)
            # 计算熵正则项
            entropy = -torch.sum(sample_weight * torch.log(sample_weight + epsilon) + (1 - sample_weight) * torch.log(1 - sample_weight + epsilon))

            loss = info_cost + self.ksp_reg * l1_regularization_kspaser + self.l1_reg * l1_regularization + self.entropy_reg * entropy
            loss.backward()

            torch.nn.utils.clip_grad_norm_([sample_weight], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                sample_weight.clamp_(0, 1)

            if show_info and step % 100 == 0:
                # print(f"step: {step} | info cost: {info_cost.item():.4f} | L1: {l1_regularization.item():.4f} | entropy: {entropy.item():.4f} | 区分度: {self.discrepancy_rate:.5f} | 目标区分度：{(1 - sample_ratio) * 100}")
                # print(f"step: {step} info loss: {loss.item():.4f} | info cost: {info_cost.item():.4f} | H2: {H2.item():.4f} | CS: {CS.item():.4f} | L1: {l1_regularization.item():.4f} | 区分度: {self.discrepancy_rate:.4f} | 目标区分度：{(1 - sample_ratio) * 100}")

                self.print_tensor_distribution(sample_weight, show_info=False)

                if l1_regularization.item() >= int(sample_num / self.amplify_ratio) and l1_regularization.item() <= sample_num and abs(self.discrepancy_rate - (1 - sample_ratio) * 100) < 0.5 or step == max_steps or self.discrepancy_rate > self.stop_sample or self.discrepancy_rate >= (1 - sample_ratio) * 100:
                    # if l1_regularization.item() > int(sample_num / self.amplify_ratio) and l1_regularization.item() < sample_num and self.discrepancy_rate > self.stop_sample  or step == max_steps:
                    print(f"Info sample过程已收敛在step: {step} | info cost: {info_cost.item():.4f} | L1: {l1_regularization.item():.4f} | entropy: {entropy.item():.4f} | 区分度: {self.discrepancy_rate:.4f} | 目标区分度：{(1 - sample_ratio) * 100}")
                    # self.print_tensor_distribution(sample_weight)
                    self.discrepancy_rate = 0
                    break

        self.H2_task_data = None
        return sample_weight

    def initialize_sample_weight(self, data_size, method="uniform",
                                 device="cuda" if torch.cuda.is_available() else "cpu"):
        if method == "uniform":
            weight = torch.rand(data_size, device=device, requires_grad=True)
            return weight
        elif method == "normal":
            return torch.randn(data_size, device=device, requires_grad=True)
        elif method == "constant":
            return torch.full((data_size,), self.weight_init, device=device, requires_grad=True)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def print_tensor_distribution(self, tensor, show_info=False):
        """
        输入一个一维张量，统计并打印在0-1范围内以不同步长的分布情况（百分比形式）

        参数:
        tensor (torch.Tensor): 一维张量

        返回:
        None
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("输入必须是一个PyTorch张量")
        if tensor.dim() != 1:
            raise ValueError("输入必须是一维张量")

        # 将张量转换为NumPy数组，确保张量在CPU上，并分离梯度
        tensor_np = tensor.detach().cpu().numpy()

        # 定义区间边界
        bins = np.array([0, 0.0001, 0.001, 0.01, 0.1] + list(np.arange(0.2, 1.3, 0.1)))
        # bins = np.array([0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,])
        counts, _ = np.histogram(tensor_np, bins)

        # 计算总数以便计算百分比
        total = counts.sum()

        # 打印统计信息（百分比形式）
        for i in range(len(bins) - 1):
            percentage = (counts[i] / total) * 100
            if show_info:
                print(f"元素在区间 [{bins[i]:.4f}, {bins[i + 1]:.4f}): {percentage:.5f}%")

            # 将0-0.001区间的百分率赋值给self.discrepancy_rate
            if bins[i] == 0 and bins[i + 1] == 0.0001:
                self.discrepancy_rate = percentage

            # if bins[i] == 0.9 and bins[i + 1] == 1.0:
            #     self.discrepancy_rate = 1 - percentage

    def get_info_cost(self, task_data, sample_data, minibatch=True):
        if minibatch:
            H2_entropy = self.get_H2_entropy(sample_data)
            CS_divergence = self.get_CS_divergence_batch(task_data, sample_data)
            info_cost = self.lamd_H2 * H2_entropy + self.lamd_CS * CS_divergence

        else:
            if self.H2_task_data is None:
                self.H2_task_data = self.get_H2_entropy(task_data)
            H2_entropy = self.get_H2_entropy(sample_data)
            CS_divergence = self.get_CS_divergence_nobatch(task_data, sample_data)
            info_cost = self.lamd_H2 * H2_entropy + self.lamd_CS * CS_divergence

        return info_cost, H2_entropy, CS_divergence

    def get_H2_entropy(self, sample_data):
        sigma = self.band_width
        kernel = self.gaussian_kernel_distance(sample_data, sigma)
        information_potential = torch.sum(kernel) / (sample_data.size(0) ** 2)
        renyis_entropy = -torch.log(information_potential + epsilon)
        return renyis_entropy

    def get_CS_divergence_batch(self, task_data, sample_data):
        sigma = self.band_width
        kernel = self.cross_gaussian_kernel_distance(task_data, sample_data, sigma)
        cross_information_potential = torch.sum(kernel) / (task_data.size(0) * sample_data.size(0))
        renyis_cross_entropy = -torch.log(cross_information_potential + epsilon)
        H2_task_data = self.get_H2_entropy(task_data)
        H2_sample_data = self.get_H2_entropy(sample_data)
        CS_divergence = 2 * renyis_cross_entropy - H2_task_data - H2_sample_data
        return CS_divergence

    def get_CS_divergence_nobatch(self, task_data, sample_data):
        sigma = self.band_width
        kernel = self.cross_gaussian_kernel_distance(task_data, sample_data, sigma)
        cross_information_potential = torch.sum(kernel) / (task_data.size(0) * sample_data.size(0))
        renyis_cross_entropy = -torch.log(cross_information_potential + epsilon)
        # H2_task_data = self.get_H2_entropy(task_data)
        H2_sample_data = self.get_H2_entropy(sample_data)
        CS_divergence = 2 * renyis_cross_entropy - self.H2_task_data - H2_sample_data
        return CS_divergence

    def gaussian_kernel_distance(self, sample_data, sigma):
        m = sample_data.size(0)
        sample_data_flat = sample_data.view(m, -1)
        distances = torch.cdist(sample_data_flat, sample_data_flat, p=2).pow(2)
        kernel_values = torch.exp(-distances / (2 * sigma ** 2))
        return kernel_values

    def cross_gaussian_kernel_distance(self, data_1, data_2, sigma=1.0):
        data_1_flat = data_1.view(data_1.size(0), -1)
        data_2_flat = data_2.view(data_2.size(0), -1)
        cross_distances = torch.cdist(data_1_flat, data_2_flat, p=2).pow(2)
        kernel_values = torch.exp(-cross_distances / (2 * sigma ** 2))
        return kernel_values

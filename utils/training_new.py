def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
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
        model.meta_begin_task(dataset)
        #########################
        model.task_counter += 1
        # model.current_task_finished = False
        #########################
        print("start training")
        if not args.inference_only:
            if t and args.enable_other_metrics:
                accs = evaluate(model, dataset, last=True)
                results[t - 1] = results[t - 1] + accs[0]
                if dataset.SETTING == 'class-il':
                    results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

            scheduler = dataset.get_scheduler(model, args) if not hasattr(model, 'scheduler') else model.scheduler

            for epoch in range(model.args.n_epochs):
                train_iter = iter(train_loader)
                data_len = None
                if not isinstance(dataset, GCLDataset):
                    data_len = len(train_loader)
                i = 0
                while True:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        #############################################
                        # model.current_task_finished = True
                        info_sample_data, info_sample_labels, info_sample_logits = model.get_info_sample(train_loader)
                        model.temp_buffer.add_data(info_sample_data, info_sample_labels, info_sample_logits)
                        model.integrate_buffers()
                        break
                    #############################################
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
                save_obj['buffer'] = deepcopy(model.main_buffer).to('cpu')

            checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
            torch.save(save_obj, checkpoint_name)

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
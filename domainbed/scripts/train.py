# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
import sys
from copy import deepcopy
from collections import defaultdict

from math import ceil
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed.datasets import get_dataloader, get_mix_dataloader
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc, swa_utils
from domainbed.lib.Logger import Logger
from domainbed.lib.swad import LossValley

import nni

def get_dataloaders(args, hparams, logger):
    # set up dataloaders
    if args.mix:
        trainloaders = [get_mix_dataloader(args.txtdir, args.dataset, args.source, "train", hparams["batch_size"]),]
        logger.info("Train size: %d" % len(trainloaders[0].dataset))
    else:
        trainloaders = [get_dataloader(args.txtdir, args.dataset, domain, "train", hparams["batch_size"]) for domain in args.source]
        for index, domain in enumerate(args.source):
            logger.info("Train %s size: %d" % (domain, len(trainloaders[index].dataset)))
    valloaders = [get_dataloader(args.txtdir, args.dataset, domain, "val", hparams["batch_size"]) for domain in args.source]
    testloaders = [get_dataloader(args.txtdir, args.dataset, domain, "test", hparams["batch_size"]) for domain in args.target]
    
    for index, domain in enumerate(args.source):
        logger.info("Val %s size: %d" % (domain, len(valloaders[index].dataset)))
    for index, domain in enumerate(args.target):
        logger.info("Test %s size: %d" % (domain, len(testloaders[index].dataset)))

    return trainloaders, valloaders, testloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--txtdir', type=str, default="/home/hanyu/dataset/txtlist")
    parser.add_argument('--dataset', type=str, default="NICO_plus")
    parser.add_argument("--gpu_id", type=int, default=3, help="gpu id")
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target", nargs="+")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument('--mix', action='store_true')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--trainable_layers_start', type=int, default=0)
    parser.add_argument('--resnet18', action='store_true')
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingLR")
    parser.add_argument('--swad', action='store_true')
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--steps', type=int, default=5000,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--hparams_config', type=str,
        help='config of hparams (fixed value, not random)')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
        help='Checkpoint every N epochs.')
    parser.add_argument('--stepval_freq', type=int, default=20,
        help='print step val every N steps.')
    parser.add_argument('--checkpoint_last', type=int, default=5,
        help='Checkpoint in the last epochs.')
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_model_best', action='store_true')
    parser.add_argument('--load_model_best', action='store_true')
    args = parser.parse_args()

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    optimized_hparams = nni.get_next_parameter()
    hparams.update(optimized_hparams)

    if args.hparams_config:
        with open(args.hparams_config) as f:
            hparams.update(json.load(f))

    # hard coding
    hparams["pretrain"] = args.pretrain
    hparams["swad"] = args.swad 
    hparams["linear_probe"] = args.linear_probe
    hparams["trainable_layers_start"] = args.trainable_layers_start
    hparams["resnet18"] = args.resnet18
    hparams["optimizer"] = args.optimizer
    hparams["scheduler"] = args.scheduler
    hparams["do_ms"] = True if args.algorithm == "MixStyle" else False
    if (args.algorithm == "MixStyle" and hparams["ms_type"] == "crossdomain") or args.optimizer == "Adahessian":
        hparams["batch_size"] = hparams["batch_size"]//2
    if args.optimizer == "Adahessian":
        hparams["lr"] = hparams["lr"]*100

    logger = Logger(args, hparams)
    
    logger.info("Environment:")
    logger.info("\t`P`ython: {}".format(sys.version.split(" ")[0]))
    logger.info("\tPyTorch: {}".format(torch.__version__))
    logger.info("\tTorchvision: {}".format(torchvision.__version__))
    logger.info("\tCUDA: {}".format(torch.version.cuda))
    logger.info("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.info("\tNumPy: {}".format(np.__version__))
    logger.info("\tPIL: {}".format(PIL.__version__))

    logger.info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.info('\t{}: {}'.format(k, v))


    logger.info('HParams:')
    for k, v in sorted(hparams.items()):
        logger.info('\t{}: {}'.format(k, v))

    os.makedirs(os.path.join(args.output_dir, Logger.get_expname(args, hparams)), exist_ok=True)

    if args.load_model_best:
        load_model_path = os.path.join(args.output_dir, Logger.get_expname(args, hparams), 'model_best_seed%d.pkl' % args.seed)
        algorithm_dict = torch.load(load_model_path, map_location="cpu")["model_dict"]
        logger.info("Load model from %s" % load_model_path)
    else:
        algorithm_dict = None
        logger.info("Do not load model")

    misc.setup_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_loaders, val_loaders, test_loaders = get_dataloaders(args, hparams, logger)

    steps_per_epoch = ceil(min([len(train_loader.dataset)/hparams['batch_size'] for train_loader in train_loaders]))
    hparams["epochs"] = ceil(args.steps/steps_per_epoch)
    logger.info("Steps per epoch: %d, number of epochs: %d, number of steps: %d" % (steps_per_epoch, hparams["epochs"], args.steps))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class((3, 224, 224), args.num_classes,
        len(args.source), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad = LossValley(hparams["n_converge"], hparams["n_tolerance"], hparams["tolerance_ratio"])

    train_minibatches_iterator = zip(*train_loaders)

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, Logger.get_expname(args, hparams), filename))

    train_accs = []
    val_accs = dict()
    test_accs = dict()
    best_val_accs = defaultdict(float)
    best_test_accs = dict()
    for step in range(args.steps):
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device)
 
        step_val_str = ""
        
        for key, val in step_vals.items():
            if key != "train_acc":
                step_val_str = step_val_str + "%s: %.3f, " % (key, val)
        train_accs.append(step_vals["train_acc"])

        if hparams["swad"]:
            swad_algorithm.update_parameters(algorithm, step=step)

        epoch = int((step+1) / steps_per_epoch)
        if (step+1) % args.stepval_freq == 0:
            logger.info("Step %d, Epoch %d, %s train acc: %.4f" % (step+1, epoch, step_val_str, np.mean(np.array(train_accs))))
            train_accs = []
            
        if (step+1) % steps_per_epoch == 0:
            algorithm.scheduler_step()
            logger.info("Next lr: %.8f" % (algorithm.get_lr()[0]))

        if ((step+1) % steps_per_epoch == 0 and (epoch % args.checkpoint_freq == 0 or epoch >= hparams["epochs"]-args.checkpoint_last)) or step == args.steps-1:
            # validation 
            logger.info("Start validation...")
            correct_overall = 0
            total_overall = 0
            val_loss_overall = 0.0
            for index, domain in enumerate(args.source):
                acc, correct, loss, loss_sum, total = misc.accuracy_and_loss(algorithm, val_loaders[index], None, device)
                correct_overall += correct
                total_overall += total
                val_loss_overall += loss_sum
                val_accs[domain] = acc
            val_accs["overall"] = correct_overall / total_overall
            for k, v in val_accs.items():
                logger.info("Val %s: %.4f" % (k, v))
            val_loss = val_loss_overall / total_overall
            logger.info("Val loss: %.4f" % val_loss)
            if hparams["swad"]:
                swad.update_and_evaluate(swad_algorithm, val_accs["overall"], val_loss)
                if swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break
                swad_algorithm = swa_utils.AveragedModel(algorithm)

            # test
            logger.info("Start testing...")
            correct_overall = 0
            total_overall = 0
            for index, domain in enumerate(args.target):
                acc, correct, total = misc.accuracy(algorithm, test_loaders[index], None, device)
                correct_overall += correct
                total_overall += total
                test_accs[domain] = acc
            test_accs["overall"] = correct_overall / total_overall
            for k, v in test_accs.items():
                logger.info("Test %s: %.4f" % (k, v))
            nni.report_intermediate_result({"default": val_accs["overall"], "test": test_accs["overall"]})

            if val_accs["overall"] > best_val_accs["overall"]:
                logger.info("New best validation acc at epoch %d!" % epoch)
                best_val_accs = deepcopy(val_accs)
                best_test_accs = deepcopy(test_accs)
                if args.save_model_best:
                    save_checkpoint('model_best_seed%d.pkl' % args.seed)
                    logger.info("Save current best model at epoch %d!" % epoch)

            if args.save_model_every_checkpoint:
                save_checkpoint('model_epoch%d_seed%d.pkl' % (epoch, args.seed))
            logger.info("")

    
    if hparams["swad"]:
        logger.info("Evaluate SWAD ...")
        swad_algorithm = swad.get_final_model()
        if not hparams["pretrain"]:
            logger.info(f"Update SWAD BN statistics for %d steps ..." % hparams["n_steps_bn"])
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, hparams["n_steps_bn"])
        correct_overall = 0
        total_overall = 0
        for index, domain in enumerate(args.target):
            acc, correct, total = misc.accuracy(swad_algorithm, test_loaders[index], None, device)
            correct_overall += correct
            total_overall += total
            best_test_accs[domain] = acc
        best_test_accs["overall"] = correct_overall / total_overall
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        logger.info(step_str)

    logger.info("Final result")
    for k, v in best_val_accs.items():
        logger.info("Best val %s: %.4f" % (k, v))
    for k, v in best_test_accs.items():
        logger.info("Best test %s: %.4f" % (k, v))

    nni.report_final_result({"default": best_val_accs["overall"], "test": best_test_accs["overall"]})

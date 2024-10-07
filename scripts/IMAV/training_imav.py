# -----------------------------------------------------------------------------#
# Copyright(C) 2021-2022 ETH Zurich, Switzerland, University of Bologna, Italy#
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE.apache.md in the top directory for details.                     #
# You may obtain a copy of the License at                                     #
#                                                                             #
#   http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# File:    training.py                                                        #
# Author:  Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
# Date:    1.09.2022                                                         #
# -----------------------------------------------------------------------------#
#
# Description:
# This script is used to train the weights of the PULP-DroNet CNN.
# You must specify the CNN architecture (dronet_dory or dronet_autotiler) and
# which dataset would you like to use:
#   - original: this dataset is composed of images from Udacity and Zurich Bicycle
#               datasets. The images have been preprocessed to a 200x200 size
#               and grayscale format to mimic the HIMAX camera format
#   - original_and_himax: this dataset adds a small set of images acquired with
#                         the HIMAX camera (on-board of the nano drone).
#                         It is used to help the network generalizing better.
# '--early_stopping': When deactivated, this script will save the trained weights
#                     (".pth" files) for all the epochs (i.e., for 100 epochs the
#                     output will be a set of 100 weights).
#                     When activated, the script will save just one set of weights
#                     (the last one, which is not necessarily the best performing one).

# TODO
# [x] consider using "class AverageMeter" for storing the loss.
# TODO (minor)
# [] GPU parallelism is not supported. Only training with 1 GPU. should we parallelize?
# [] alpha and beta variables for hard mining are global. should be local


# essentials
import os
import sys
import argparse
from unicodedata import name
import numpy as np
import shutil
from os.path import join
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import cv2

# torch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# PULP-dronet
# from models import Dataset
from utility import EarlyStopping, init_weights
from utility import DronetDatasetV3, CustomDataset
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss_v3
from utility import AverageMeter
from utility import write_log
# from augmentation import himax_augment

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy import stats

# nemo
# sys.path.append('/home/lamberti/work/nemo') # if you want to use your custom installation (git clone) instead of pip version
import nemo

import wandb
import random


def create_parser(cfg):
    parser = argparse.ArgumentParser(description="PyTorch PULP-DroNet Training")
    parser.add_argument(
        "-d", "--data_path", help="path to dataset", default=cfg.data_path
    )
    parser.add_argument("--data_path_testing", help="path to dataset")
    parser.add_argument(
        "--partial_training",
        default="collision",
        choices=[None, "classification", "regression", "edge", "yaw", "collision"],
        help="leave None to train on classification+regression, \
                        select classification to train just on collision, select\
                        regression to train just on yaw-rate",
    )
    parser.add_argument(
        "--image_size", default="162x162", choices=["324x324", "162x162"]
    )
    parser.add_argument(
        "--yaw_rate_as_classification",
        action="store_true",
        help="transform the regression of yaw_rate into a \
                        classification problem: [left, straingt, right]",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default=cfg.model_name,
        help="model name that is created when training",
    )
    parser.add_argument(
        "-w",
        "--model_weights",
        default=cfg.model_weights,
        help="path to the weights for resuming training(.pth file)",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default=cfg.arch,
        choices=[
            "dronet_dory",
            "dronet_autotiler",
            "dronet_dory_no_residuals",
            "dronet_imav",
        ],
        help="select the NN architecture backbone:",
    )
    parser.add_argument(
        "--block_type",
        action="store",
        choices=["ResBlock", "Depthwise", "Inverted"],
        default="ResBlock",
    )
    parser.add_argument(
        "--depth_mult",
        default=cfg.depth_mult,
        type=float,
        help="depth multiplier that scales number of channels",
    )
    parser.add_argument(
        "--gpu",
        help="which gpu to use. Just one at" "the time is supported",
        default=cfg.gpu,
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=cfg.workers,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=cfg.epochs,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=cfg.training_batch_size,
        type=int,
        metavar="N",
        help="mini-batch size (default: 32), this is the total "
        "batch size of all GPUs",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=cfg.learning_rate,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--lr_decay",
        default=cfg.lr_decay,
        type=float,
        help="learning rate decay (default: 1e-5)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        default=cfg.checkpoint_path,
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: checkpoints)",
    )
    parser.add_argument(
        "--logs_dir",
        default=cfg.logs_dir,
        type=str,
        metavar="PATH",
        help="path to save log files",
    )
    parser.add_argument(
        "--hard_mining_train",
        default=cfg.hard_mining_train,
        type=bool,
        help="do training with hard mining",
    )
    parser.add_argument(
        "--early_stopping",
        default=cfg.early_stopping,
        type=bool,
        help="early stopping at training time, with (patience,delta) parameters",
    )
    parser.add_argument(
        "--patience",
        default=cfg.patience,
        type=int,
        help="patience of early stopping at training time, value in epochs",
    )
    parser.add_argument(
        "--delta",
        default=cfg.delta,
        type=float,
        help="max delta value for early stopping at training time",
    )
    parser.add_argument(
        "--resume_training",
        default=cfg.resume_training,
        type=bool,
        metavar="PATH",
        help="want to resume training?",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose prints on")
    parser.add_argument("--test", action="store_true", help="skip train")
    parser.add_argument("--true_data", action="store_true", help="")
    ## rebuttal experiments
    parser.add_argument(
        "--aug",
        default=cfg.aug,
        choices=[
            "all",
            "noise",
            "blur",
            "exposure",
            "blur_exposure",
            "blur_noise",
            "noise_exposure",
            "none",
        ],
        help="select which augmentation to apply",
    )
    # parser.add_argument('-s', '--dataset', default=cfg.dataset,
    #                     choices=['imav', 'dronet_v2'],
    #                     help='training and testing dataset')
    return parser


# Global variables
device = torch.device("cpu")
alpha = 1.0
beta = 0.0
early_stop_checkpoint = "checkpoint.pth"

# change working directory to the folder of the project
working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)
print("\nworking directory:", working_dir, "\n")


def custom_acc_collision(y_true, output, device):

    target_c = y_true
    y_pred = output[2]

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    acc = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    pred = y_pred >= 0.5
    truth = target_c.squeeze() >= 0.5
    tp += pred.mul(truth).sum(0).float()
    tn += (~pred).mul(~truth).sum(0).float()
    fp += pred.mul(~truth).sum(0).float()
    fn += (~pred).mul(truth).sum(0).float()
    acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()

    return acc


# yaw_rate classification
def custom_ce_collision(y_true, y_pred, device):

    target_c = y_true
    input_c = y_pred[2]
    loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction="mean")

    output_BCE_left = loss_BCE(input_c[:, 0], target_c[:, 0]).to(device)
    output_BCE_cent = loss_BCE(input_c[:, 1], target_c[:, 1]).to(device)
    output_BCE_right = loss_BCE(input_c[:, 2], target_c[:, 2]).to(device)

    return output_BCE_left + output_BCE_cent + output_BCE_right


# Labels: collision = 1
def custom_loss(y_true, y_pred, device, partial_training="classification"):

    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)

    # collision
    target_c = y_true[:, 1].squeeze()
    input_c = y_pred[1]
    loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction="mean")
    output_BCE = loss_BCE(input_c, target_c).to(device)
    # loss formula
    if partial_training == "classification":
        Loss = output_BCE

    return Loss

# validate and test function
def validate(
    test_set,
    net,
    data_loader,
    logs_dir,
    df_valid,
    epoch,
    device,
    wandb,
    ultrasound
):
    if test_set == "valid":
        dataset_string = "Val"
        prefix = "valid"
    elif test_set == "testing":
        dataset_string = "Test_without_smoke"
        prefix = "test"
    elif test_set =="testing_2":
        dataset_string = "Test_with_smoke"
        prefix = "test_2"

    net.eval()

    ce_collision_valid = AverageMeter("BCE_CL", ":.4f")  # Collision
    acc_collision_valid = AverageMeter("ACC_CL", ":.3f")
    loss_valid = AverageMeter("Loss", ":.4f")  # Loss
    
    for obj in [ce_collision_valid, acc_collision_valid, loss_valid]:
        obj.reset()

    pred_stat = []
    ground_truth_stat = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            inputs, labels, distance = data[0].to(device), data[1].to(device), data[2].to(device)
            
            if ultrasound:
                outputs = net(inputs, distance)
            else:
                outputs = net(inputs)
            
            ce_collision = custom_ce_collision(labels, outputs, device)
            acc_collision = custom_acc_collision(labels, outputs, device)
            
            loss = ce_collision
            
            ce_collision_valid.update(ce_collision.item())
            acc_collision_valid.update(acc_collision.item())
            loss_valid.update(loss.item())
            
            pred_stat.extend(outputs[2].cpu().numpy().flatten())
            ground_truth_stat.extend(labels.cpu().numpy().flatten())

        # log the results to wandb
        if test_set == "valid":
            wandb.log(
                {
                    f"Loss/{dataset_string}": loss_valid.avg,
                    f"Acc/{dataset_string}": acc_collision_valid.avg * 100,
                    "Epoch": epoch,
                }
            )
        else:
            wandb.log(
                {
                    f"Loss/{dataset_string}": loss_valid.avg,
                    f"Acc/{dataset_string}": acc_collision_valid.avg * 100,
                }
            )
        
        pred_stat = np.array(pred_stat)
        ground_truth_stat = np.array(ground_truth_stat)
        
        r2 = r2_score(ground_truth_stat, pred_stat)
        pearson = stats.pearsonr(ground_truth_stat, pred_stat)[0]
        mae = mean_absolute_error(ground_truth_stat, pred_stat)
        mse = mean_squared_error(ground_truth_stat, pred_stat)
        
        wandb.log(
            {
                f"R2/{dataset_string}": r2,
                f"Pearson/{dataset_string}": pearson,
                f"MAE/{dataset_string}": mae,
                f"MSE/{dataset_string}": mse,
            }
        )
    
    print(
        dataset_string + "\tACC COLL: %.4f" % acc_collision_valid.avg,
        "\t||",
        "\tCE COLL:  %.4f" % float(ce_collision_valid.avg),
    )

    log_str = (
        "Valid [{0}][{1}/{2}]\t"
        "acc_collision (avg) {acc_collision.val:.3f} ({acc_collision.avg:.3f})\t"
        "ce_collision (avg) {ce_collision.val:.3f} ({ce_collision.avg:.3f})\t".format(
            epoch,
            batch_idx,
            len(data_loader),
            loss=loss_valid,
            acc_collision=acc_collision_valid,
            ce_collision=ce_collision_valid,
        )
    )

    write_log(logs_dir, log_str, prefix=prefix, should_print=False, mode="a", end="\n")

    return df_valid, ce_collision_valid.avg



################################################################################
# MAIN
################################################################################


def main():
    # parse arguments
    global args
    from config import cfg  # load configuration with all default values

    parser = create_parser(cfg)
    args = parser.parse_args()
    save_checkpoints_path = args.checkpoint_path
    model_name = args.model_name
    model_parameters_path = "model"
    print("Model name:", model_name)

    # select device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)


    if not args.data_path_testing:
        args.data_path_testing = args.data_path
    print("Training and Validation set paths:", args.data_path)
    print(
        "Testing set path (you should select the non augmented dataset):",
        args.data_path_testing,
    )
    
    for run in range(1, 4):
        # run is used to select the configuration of the run
        # train_flag indicates if we want to train the model
        # ultrasound indicates if we want to use the ultrasonic data
        # tethys indicates if we want to use tethys for training/testing
        # train_room indicates if we want to use the train room or the test room
        # mid_fusion indicates if we want to use the mid fusion model or the late fusion model
        
        # run = 2
        if run == 1:
            train_flag = True
            ultrasound = False
            tethys = True
            train_room = True
            mid_fusion= False
            model_name_pth = ""
        elif run == 2:
            train_flag = True
            ultrasound = True
            tethys = True
            train_room = True
            mid_fusion= True
            model_name_pth = ""
        else:
            train_flag = True
            ultrasound = True
            tethys = True
            train_room = True
            mid_fusion= False
            model_name_pth = ""
        
        print(f"Run {run}")
        print(f"train_flag = {train_flag}, ultrasound = {ultrasound}, tethys = {tethys}, train_room = {train_room}, mid_fusion = {mid_fusion}")
        
        seeds_list = [0, 42, 123, 666, 999]
        for i in seeds_list:
            seed_value = i
            print(f"seed = {seed_value}")
            np.random.seed(seed_value)
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            wandb.login()

            if tethys:
                # config is used to set the hyperparameters
                config = dict(
                    epochs=100,
                    train_batch=1024,
                    val_batch=256,
                    test_batch=128,
                    learning_rate=0.001,
                )
                
                train = pd.read_pickle("/data/df_train.pkl")
                if train_room:
                    test = pd.read_pickle("/data/df_test.pkl")
                else:
                    test = pd.read_pickle("/data/df_test_new_room_2.pkl")
                
            else:
                # config is used to set the hyperparameters
                config = dict(
                    epochs=100,
                    train_batch=64,
                    val_batch=32,
                    test_batch=32,
                    learning_rate=0.001,
                )
                
                train = pd.read_pickle("data/df_train.pkl")
                if train_room:
                    test = pd.read_pickle("data/df_test.pkl")
                else:
                    test = pd.read_pickle("data/df_test_new_room.pkl")

            if ultrasound:
                if mid_fusion:
                    model_name_pth = f"{seed_value}_IMAV_Mid_Fusion_ultrasound"
                else:
                    model_name_pth = f"{seed_value}_IMAV_Late_Fusion_ultrasound"
            else:
                model_name_pth = f"{seed_value}_IMAV_no_ultrasound"
            print(f"Model name: {model_name_pth}")
            
            # wandb initialization
            # wandb.init(config=config, project="master_thesis", name=model_name_pth)
            
            # data preprocessing
            input_transform = transforms.Compose(
                [
                    transforms.Resize((162, 162)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

            # dataset creation
            dataset = CustomDataset(
                data=train,
                input_transform=input_transform,
                train=True,
            )

            test_dataset = CustomDataset(
                data=test,
                input_transform=input_transform,
                train=False,
                smoke=False,
            )
            
            test_dataset_smoke = CustomDataset(
                data=test,
                input_transform=input_transform,
                train=False,
                smoke=True,
            )

            # dataset splitting
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # dataloader creation
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["train_batch"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            validation_loader = DataLoader(
                val_dataset,
                batch_size=config["val_batch"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["test_batch"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            
            test_loader_smoke = DataLoader(
                test_dataset_smoke,
                batch_size=config["test_batch"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            if args.arch == "dronet_imav":
                from model.dronet_imav import dronet_imav, ResBlock, dronet_imav_v2, dronet_imav_v3
            else:
                raise ValueError(
                    "Doublecheck the architecture that you are trying to use.\
                                    Select one between dronet_dory and dronet_autotiler"
                )
            print("You are using the", args.arch, "CNN architecture\n")

            # get image size and define stride stages
            image_size = list(np.squeeze(next(iter(test_dataset))[0]).shape)
            stride = [2, 2, 2, 2, 2]
            if args.image_size == "162x162":
                fc_in_size = [5, 5]
            else:
                fc_in_size = [11, 11]

            # select the CNN model
            print("You are using a depth multiplier of", args.depth_mult, "for PULP-Dronet")
            if args.block_type == "ResBlock":
                if ultrasound:
                    if mid_fusion:
                        net = dronet_imav_v2(
                            depth_mult=args.depth_mult,
                            stride=stride,
                            image_size=image_size,
                            fc_in_size=fc_in_size,
                            block_class=ResBlock,
                            outputs=7,
                        )
                    else:
                        net = dronet_imav_v3(
                            depth_mult=args.depth_mult,
                            stride=stride,
                            image_size=image_size,
                            fc_in_size=fc_in_size,
                            block_class=ResBlock,
                            outputs=7,
                        )
                else:
                    net = dronet_imav(
                        depth_mult=args.depth_mult,
                        stride=stride,
                        image_size=image_size,
                        fc_in_size=fc_in_size,
                        block_class=ResBlock,
                        outputs=7,
                    )
            
            # initialize weights and biases for training
            if args.resume_training or args.test:
                if os.path.isfile(args.model_weights):
                    if torch.cuda.is_available():
                        checkpoint = torch.load(args.model_weights, map_location=device)
                        print("loaded checkpoint on cuda")
                    else:
                        checkpoint = torch.load(args.model_weights, map_location="cpu")
                        print("CUDA not available: loaded checkpoint on cpu")
                    if "state_dict" in checkpoint:
                        checkpoint = checkpoint["state_dict"]
                    else:
                        print(
                            "Warning: failed to find the ["
                            "state_dict"
                            "] inside the checkpoint. I will try to open it anyways."
                        )
                    net.load_state_dict(checkpoint)
                    print("Loaded weights successfully")
                else:
                    raise RuntimeError(
                        "Failed to open checkpoint. provide a checkpoint.pth.tar file"
                    )
            else:
                net.apply(init_weights)

            net.to(device)
            
            # summary
            # input_size = tuple(
            #     [
            #         1,
            #         1,
            #     ]
            #     + image_size,
            # )
            # net(torch.unsqueeze(next(iter(train_dataset))[0],0).to(device)) # debug
            # summary(net, input_size=input_size)
            

            # uncomment to get the number of parameters and MACs
            # from ptflops import get_model_complexity_info
            # n_mac, n_params = get_model_complexity_info(net, (1, 162, 162), as_strings=True, backend='pytorch', print_per_layer_stat=True)
            # print('FLOPs: ', n_mac)
            # print('Params: ', n_params)

            # initialize the optimizer for training
            optimizer = optim.Adam(
                net.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.lr_decay,
                amsgrad=False,
            )

            # initialize the early_stopping object
            if args.early_stopping:
                early_stopping = EarlyStopping(
                    patience=args.patience,
                    delta=args.delta,
                    verbose=True,
                    # path=save_checkpoints_path + model_name + "/" + early_stop_checkpoint,
                    path = f"/data/{model_name_pth}.pth", # tethys
                    # path = "data/IMAV.pth", # local
                )

            if args.verbose:
                # Print model's state_dict
                print("Model's state_dict:")
                for param_tensor in net.state_dict():
                    print(param_tensor, "\t\t\t", net.state_dict()[param_tensor].size())
                # Print optimizer's state_dict
                print("Optimizer's state_dict:")
                for var_name in optimizer.state_dict():
                    print(var_name, "\t", optimizer.state_dict()[var_name])

            # create training directory
            training_dir = join(os.path.dirname(__file__), "training")
            training_model_dir = join(training_dir, model_name)
            logs_dir = join(training_model_dir, args.logs_dir)
            tensorboard_dir = join(
                training_model_dir, "tensorboard_" + datetime.now().strftime("%b%d_%H:%M:%S")
            )
            checkpoint_dir = join(training_model_dir, "checkpoint")

            os.makedirs(logs_dir, exist_ok=True)
            print("Logs directory: ", logs_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            print("Tensorboard directory: ", tensorboard_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            print("Checkpoints directory: ", checkpoint_dir)

            # write the training/validation/testing paths to logs. By doing so we keep track of which dataset we use (augemented VS not augmented)
            write_log(
                logs_dir,
                "Training data path:\t" + args.data_path,
                prefix="train",
                should_print=False,
                mode="a",
                end="\n",
            )
            write_log(
                logs_dir,
                "Validation data path:\t" + args.data_path,
                prefix="valid",
                should_print=False,
                mode="a",
                end="\n",
            )
            write_log(
                logs_dir,
                "Testing data path:\t" + args.data_path_testing,
                prefix="test",
                should_print=False,
                mode="a",
                end="\n",
            )

            # logging utils
            ce_collision_train = AverageMeter("BCE_CL", ":.4f")  # Collision
            acc_collision_train = AverageMeter("ACC_CL", ":.3f")
            loss_train = AverageMeter("Loss", ":.4f")  # Loss
            # dataframes for csv files
            df_train = pd.DataFrame(columns=["Epoch", "BCE_CL", "ACC_CL", "Loss"])
            df_valid = pd.DataFrame(columns=["Epoch", "BCE_CL", "ACC_CL", "Loss"])
            df_test = pd.DataFrame(columns=["Epoch", "BCE_CL", "ACC_CL", "Loss"])
            
            ############################################################################
            # Train Loop Starts Here
            ############################################################################
            if train_flag:
                print("Training started")
                for epoch in range(args.epochs + 1):
                    for obj in [
                        ce_collision_train,
                        acc_collision_train,
                        loss_train,
                    ]: obj.reset()
                    
                    #### TRAINING ####
                    print("Epoch: %d/%d" % (epoch, args.epochs))
                    net.train()
                    with tqdm(total=len(train_loader), desc="Train", disable=not True) as t:
                        for batch_idx, data in enumerate(train_loader):
                            
                            inputs, labels, distance = data[0].to(device), data[1].to(device), data[2].to(device)
                            optimizer.zero_grad()
                            if ultrasound:
                                # outputs = [edge, no_edge, corner, yaw_target, left_pcoll, center_pcoll. right_pcoll]
                                outputs = net(inputs, distance)
                            else:
                                # outputs = [edge, no_edge, corner, yaw_target, left_pcoll, center_pcoll. right_pcoll]
                                outputs = net(inputs)

                            ce_collision = custom_ce_collision(labels, outputs, device)
                            acc_collision = custom_acc_collision(labels, outputs, device)

                            loss = ce_collision

                            loss.backward()
                            optimizer.step()
                            
                            # store values
                            ce_collision_train.update(ce_collision.item())
                            acc_collision_train.update(acc_collision.item())
                            loss_train.update(loss.item())

                            t.set_postfix(
                                {"loss": loss_train.avg, "acc_coll": acc_collision_train.avg}
                            )
                            
                            wandb.log(
                                {
                                    "Loss/Train": loss_train.avg,
                                    "Acc/Train": acc_collision_train.avg,
                                    "Epoch": epoch,
                                }
                            )
                            t.update(1)

                    log_str = (
                        "Train [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "acc_collision {acc_collision.val:.3f} ({acc_collision.avg:.3f})\t"
                        "ce_collision {ce_collision.val:.3f} ({ce_collision.avg:.3f})\t".format(
                            epoch,
                            batch_idx,
                            len(validation_loader),
                            loss=loss_train,
                            acc_collision=acc_collision_train,
                            ce_collision=ce_collision_train,
                        )
                    )

                    write_log(
                        logs_dir,
                        log_str,
                        prefix="train",
                        should_print=False,
                        mode="a",
                        end="\n",
                    )

                    #### VALIDATION ####
                    df_valid, valid_loss = validate(
                        "valid",
                        net,
                        validation_loader,
                        logs_dir,
                        df_valid,
                        epoch,
                        device,
                        wandb,
                        ultrasound
                    )


                    # early_stopping needs the validation loss to check if it has decresed,
                    # and if it has, it will make a checkpoint of the current model
                    if args.early_stopping:
                        early_stopping(valid_loss, net)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                print("Training Finished")

            #### Testing Set ####
            if tethys:
                net.load_state_dict(torch.load(f"/data/{model_name_pth}.pth", map_location=device))
            else:
                net.load_state_dict(torch.load(f"data/{model_name_pth}.pth", map_location=torch.device("cpu")))
            print("Testing started")
            
            # test the model
            validate(
                "testing",
                net,
                test_loader,
                logs_dir,
                df_test,
                100,
                device,
                wandb,
                ultrasound
            )
            
            validate(
                "testing_2",
                net,
                test_loader_smoke,
                logs_dir,
                df_test,
                100,
                device,
                wandb,
                ultrasound
            )
            
            wandb.finish()


if __name__ == "__main__":
    main()

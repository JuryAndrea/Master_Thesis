#
import torch
from torch import nn

from torchsummary import summary
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import numpy as np
import random

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# used for MACs and number of parameters calculation
# from ptflops import get_model_complexity_info

from scipy import stats

import pandas as pd

import wandb

from tqdm import tqdm

from training_models import MkModel, MkModel_v2, MkModel_v3, MkModel_v4, MkModel_v5, MkModel_v6, MkModel_v7

from utilities import CustomDataset, inverse_huber_loss, perform_thresholding

# test function
def test_model(model, test_dataloader, ultrasound, criterion, device, wandb, title):
    
    model.eval()

    acc = 0
    corr_coll_preds = 0
    tot_coll_preds = 0


    with torch.no_grad():
        for test_batch in test_dataloader:
            test_images, test_ground_truth, distance, labels = test_batch

            test_images = test_images.to(device)
            test_ground_truth = test_ground_truth.to(device)
            distance = distance.to(device)
            labels = labels.to(device)


            left_prob = np.zeros(labels.size(0))
            center_prob = np.zeros(labels.size(0))
            right_prob = np.zeros(labels.size(0))
            
            corr_coll_preds += (left_prob == labels[:, 0].cpu().numpy()).sum()
            corr_coll_preds += (center_prob == labels[:, 1].cpu().numpy()).sum()
            corr_coll_preds += (right_prob == labels[:, 2].cpu().numpy()).sum()

            tot_coll_preds += labels.size(0) * 3


    if tot_coll_preds > 0:
        acc = corr_coll_preds / tot_coll_preds * 100
        print(f"Accuracy: {acc}")




if __name__ == "__main__":

    for run in range(1, 3):
        # run is used to select the configuration (train, ultrasound, tethys, train_room, model_name)
        # run = 2
        if run == 1:
            train = False
            ultrasound = False
            tethys = False
            train_room = False
            model_name = ""
        else:
            train = False
            ultrasound = True
            tethys = False
            train_room = False
            model_name = ""

        print(f"Run = {run}")
        print(f"train = {train}, ultrasound = {ultrasound}, tethys = {tethys}, train_room = {train_room}")
        # seeds_list = [0, 42, 123, 666, 999]
        seeds_list = [42]
        concat_point = None
        
        for seed in seeds_list:
            seed_value = seed
            print(f"Seed = {seed_value}")
            np.random.seed(seed_value)
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            wandb.login()
            
            if tethys:
                # config is used to set the hyperparameters and logging on wandb
                config = dict(
                    epochs=100,
                    train_batch=1024,
                    val_batch=256,
                    test_batch=128,
                    learning_rate=0.001,
                )
                
                if train_room:
                    df_test = pd.read_pickle("/data/df_test.pkl")
                else:
                    df_test = pd.read_pickle("/data/df_test_new_room.pkl")
                
            else:
                # config is used to set the hyperparameters and logging on wandb
                config = dict(
                    epochs=100,
                    train_batch=64,
                    val_batch=32,
                    test_batch=32,
                    learning_rate=0.001,
                )
                
                if train_room:
                    df_test = pd.read_pickle("data/df_test.pkl")
                else:
                    df_test = pd.read_pickle("data/df_test_new_room.pkl")

            if ultrasound:
                model_name = f"{seed_value}_FCNN_ultrasound"
            else:
                model_name = f"{seed_value}_FCNN_no_ultrasound"
                

            device = (
                "cuda"
                if torch.cuda.is_available()
                # else "mps" if torch.backends.mps.is_available() else "cpu"
                else "cpu"
            )
            print(f"Using device: {device}")

            # data transforms
            input_transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

            ground_truth_transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

            # test datasets
            test_dataset = CustomDataset(
                data=df_test,
                input_transform=input_transform,
                ground_truth_transform=ground_truth_transform,
                model_name="fcnn",
                train=False,
                smoke=False,
            )
            
            test_dataset_smoke = CustomDataset(
                data=df_test,
                input_transform=input_transform,
                ground_truth_transform=ground_truth_transform,
                model_name="fcnn",
                train=False,
                smoke=True,
            )

            # test dataloaders
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config["test_batch"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            test_dataloader_smoke = DataLoader(
                test_dataset_smoke,
                batch_size=config["test_batch"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )


            if ultrasound:
                # all possible concatenation point for the ultrasonic scalar input
                model = MkModel_v5().to(device)
                # if concat_point == 2:
                #     model = MkModel_v2().to(device)
                # elif concat_point == 3:
                #     model = MkModel_v3().to(device)
                # elif concat_point == 4:
                #     model = MkModel_v4().to(device)
                # elif concat_point == 5:
                #     model = MkModel_v5().to(device)
                # elif concat_point == 6:
                #     model = MkModel_v6().to(device)
                # else:
                #     model = MkModel_v7().to(device)
            else:
                model = MkModel().to(device)


            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            criterion = torch.nn.L1Loss()

            num_epochs = config["epochs"]

            # initialize wandb
            # wandb.init(config=config, project="master_thesis", name=model_name)
            
            if tethys:
                model.load_state_dict(torch.load(f"/data/{model_name}.pth", map_location=device)) # tethys
            else:
                model.load_state_dict(torch.load(f"data/{model_name}.pth", map_location=device)) # local

            # model's test function
            test_model(model, test_dataloader, ultrasound, criterion, device, wandb, "_without_smoke")
            test_model(model, test_dataloader_smoke, ultrasound, criterion, device, wandb, "_with_smoke")
            
            wandb.finish()

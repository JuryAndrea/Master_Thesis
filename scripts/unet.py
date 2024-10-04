#
import torch
from torch import nn

from torchsummary import summary
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
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

from training_models import UNet, UNet_v2

from utilities import CustomDataset, perform_thresholding

# test function
def test_model(model, test_dataloader, ultrasound, criterion, device, wandb, title):
    model.eval()
    
    num_batches = 0

    test_loss = 0

    test_pred_stat = []
    test_ground_truth_stat = []

    acc = 0
    corr_coll_preds = 0
    tot_coll_preds = 0

    with torch.no_grad():
        for test_batch in test_dataloader:
            test_images, test_ground_truth, test_distance, labels = test_batch

            test_images = test_images.to(device)
            test_ground_truth = test_ground_truth.to(device)
            labels = labels.to(device)
            test_distance = test_distance.to(device)

            if ultrasound:
                test_pred = model(test_images, test_distance).to(device)
            else:
                test_pred = model(test_images).to(device)

            test_ground_truth_stat.extend(test_ground_truth.cpu().numpy().flatten())
            test_pred_stat.extend(test_pred.cpu().numpy().flatten())

            loss = criterion(test_pred, test_ground_truth)
            
            test_pred = test_pred.squeeze(1)
            # split the image in 3 parts
            left = test_pred.cpu().detach()[:, :, :160 // 3]
            center = test_pred.cpu().detach()[:, :, 160 // 3 : 2 * 160 // 3]
            right = test_pred.cpu().detach()[:, :, 2 * 160 // 3 :]

            # check if at least 10% of the pixels are above the threshold of 2 meters
            left_prob = perform_thresholding(left)
            center_prob = perform_thresholding(center)
            right_prob = perform_thresholding(right)

            # check predictions against true labels
            corr_coll_preds += (left_prob == labels[:, 0].cpu().numpy()).sum()
            corr_coll_preds += (center_prob == labels[:, 1].cpu().numpy()).sum()
            corr_coll_preds += (right_prob == labels[:, 2].cpu().numpy()).sum()

            tot_coll_preds += 3 * labels.size(0)

            test_loss += loss.item()

            num_batches += 1

    # log the results on wandb
    if num_batches > 0:
        test_loss /= num_batches
        wandb.log({f"Loss/Test{title}": test_loss})
    
    if tot_coll_preds > 0:
        acc = (corr_coll_preds / tot_coll_preds) * 100
        wandb.log({f"Acc/Test{title}": acc})

    test_pred_stat = np.array(test_pred_stat)
    test_ground_truth_stat = np.array(test_ground_truth_stat)

    r2_score_test = r2_score(test_ground_truth_stat, test_pred_stat)
    pearson_test = stats.pearsonr(test_ground_truth_stat, test_pred_stat)[0]
    mae_test = mean_absolute_error(test_ground_truth_stat, test_pred_stat)
    mse_test = mean_squared_error(test_ground_truth_stat, test_pred_stat)

    wandb.log(
        {
            f"R2/Test{title}": r2_score_test,
            f"MAE/Test{title}": mae_test,
            f"MSE/Test{title}": mse_test,
            f"Pearson/Test{title}": pearson_test,
        }
    )


if __name__ == "__main__":
    
    wandb.login()

    train = True
    ultrasound = True
    tethys = True
    train_room = True
    model_name = ""
    
    print(f"train = {train}, ultrasound = {ultrasound}, tethys = {tethys}, train_room = {train_room}")
    seeds_list = [0, 42, 123, 666, 999]

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

        if tethys:
            # config is used to set the hyperparameters of the model and loggin them on wandb
            config = dict(
            epochs=100,
            train_batch=1024,
            val_batch=512,
            test_batch=128,
            learning_rate=0.001,
            )
            
            df_train = pd.read_pickle("/data/df_train.pkl")
            if train_room:
                df_test = pd.read_pickle("/data/df_test.pkl")
            else:
                df_test = pd.read_pickle("/data/df_test_new_room.pkl")
        else:
            # config is used to set the hyperparameters of the model and loggin them on wandb
            config = dict(
            epochs=100,
            train_batch=64,
            val_batch=32,
            test_batch=32,
            learning_rate=0.001,
            )
            
            df_train = pd.read_pickle("data/df_train.pkl")
            if train_room:
                df_test = pd.read_pickle("data/df_test.pkl")
            else:
                df_test = pd.read_pickle("data/df_test_new_room.pkl")
        
        if ultrasound:
            model_name = f"{seed_value}_UNet_ultrasound"
        else:
            model_name = f"{seed_value}_UNet_no_ultrasound"
        print(f"model name is {model_name}")

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {device}")


        # dat preparation
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

        # datasets creation
        dataset = CustomDataset(
            data=df_train,
            input_transform=input_transform,
            ground_truth_transform=ground_truth_transform,
            model_name="unet",
            train=True,
        )

        test_dataset = CustomDataset(
            data=df_test,
            input_transform=input_transform,
            ground_truth_transform=ground_truth_transform,
            model_name="unet",
            train=False,
            smoke=False,
        )
        
        test_dataset_smoke = CustomDataset(
            data=df_test,
            input_transform=input_transform,
            ground_truth_transform=ground_truth_transform,
            model_name="unet",
            train=False,
            smoke=True,
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # dataloaders creation
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["train_batch"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["val_batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
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
            model = UNet_v2().to(device)
        else:
            model = UNet().to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        criterion = torch.nn.L1Loss()

        num_epochs = config["epochs"]

        # wandb initialization
        # wandb.init(config=config, project="master_thesis", name=model_name)

        # early stopping
        patience = 10
        counter = 0
        best_val_loss = np.inf

        # Training
        if train:
            for epoch in tqdm(range(num_epochs)):

                model.train()

                train_loss = 0
                num_batches = 0

                for batch in train_dataloader:

                    train_images, ground_truth_images, train_distance, _ = batch

                    train_images = train_images.to(device)
                    ground_truth_images = ground_truth_images.to(device)
                    train_distance = train_distance.to(device)

                    optimizer.zero_grad()

                    if ultrasound:
                        pred = model(train_images, train_distance).to(device)
                    else:
                        pred = model(train_images).to(device)

                    # pred.shape = (32, 1, 20, 20)
                    # ground_truth_images.shape = (32, 1, 20, 20)
                    loss = criterion(pred, ground_truth_images)

                    loss.backward()

                    optimizer.step()

                    train_loss += loss.item()

                    num_batches += 1

                if num_batches > 0:
                    train_loss /= num_batches
                    wandb.log({"Loss/Train": train_loss, "Epoch": epoch})

                # Validation
                model.eval()
                val_loss = 0
                val_pred_stat = []
                val_ground_truth_stat = []

                with torch.no_grad():

                    num_batches = 0

                    for idx, val_batch in enumerate(val_dataloader):

                        val_image, val_ground_truth, val_distance, _ = val_batch

                        val_image = val_image.to(device)
                        val_ground_truth = val_ground_truth.to(device)
                        val_distance = val_distance.to(device)

                        if ultrasound:
                            val_pred = model(val_image, val_distance).to(device)
                        else:
                            val_pred = model(val_image).to(device)

                        val_pred_stat.extend(val_pred.cpu().numpy().flatten())
                        val_ground_truth_stat.extend(val_ground_truth.cpu().numpy().flatten())

                        loss = criterion(val_pred, val_ground_truth)

                        val_loss += loss.item()
                        num_batches += 1

                    if num_batches > 0:
                        val_loss /= num_batches
                        wandb.log({"Loss/Val": val_loss, "Epoch": epoch})

                    if counter > patience:
                        print("Early stopping")
                        break
                    else:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            counter = 0
                            torch.save(model.state_dict(), f"/data/{model_name}.pth")
                        else:
                            counter += 1

                    # Metrics calculation and logging on wandb
                    val_pred_stat = np.array(val_pred_stat)
                    val_ground_truth_stat = np.array(val_ground_truth_stat)

                    r2_score_ = r2_score(val_ground_truth_stat, val_pred_stat)
                    pearson_ = stats.pearsonr(val_ground_truth_stat, val_pred_stat)[0]
                    mae_ = mean_absolute_error(val_ground_truth_stat, val_pred_stat)
                    mse_ = mean_squared_error(val_ground_truth_stat, val_pred_stat)

                    wandb.log(
                        {
                            "R2/Val": r2_score_,
                            "MAE/Val": mae_,
                            "MSE/Val": mse_,
                            "Pearson/Val": pearson_,
                            "Epoch": epoch,
                        }
                    )

                    if epoch % 10 == 0:
                        print(
                            f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}"
                        )

                        print(
                            f"Epoch [{epoch}/{num_epochs}] | R2: {r2_score_:.3f} | MAE: {mae_:.3f} | MSE: {mse_:.3f} | Pearson: {pearson_:.3f}"
                        )

        if tethys:
            model.load_state_dict(torch.load(f"/data/{model_name}.pth", map_location=device))
        else:
            model.load_state_dict(torch.load(f"data/{model_name}.pth", map_location=device))

        # test
        test_model(model, test_dataloader, ultrasound, criterion, device, wandb, "_without_smoke")
        test_model(model, test_dataloader_smoke, ultrasound, criterion, device, wandb, "_with_smoke")
        
        wandb.finish()

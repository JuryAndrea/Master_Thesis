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
    
    test_loss = 0

    test_pred_stat = []
    test_ground_truth_stat = []

    acc = 0
    corr_coll_preds = 0
    tot_coll_preds = 0

    num_batches = 0

    with torch.no_grad():
        for test_batch in test_dataloader:
            test_images, test_ground_truth, distance, labels = test_batch

            test_images = test_images.to(device)
            test_ground_truth = test_ground_truth.to(device)
            distance = distance.to(device)
            labels = labels.to(device)

            if ultrasound:
                test_pred = model(test_images, distance).to(device)
            else:
                test_pred = model(test_images).to(device)

            test_ground_truth_stat.extend(test_ground_truth.cpu().numpy().flatten())
            test_pred_stat.extend(test_pred.cpu().numpy().flatten())

            loss = criterion(test_pred, test_ground_truth.squeeze(1))

            # split the image in 3 parts
            left = test_pred.cpu().detach()[:, :, :20 // 3]
            center = test_pred.cpu().detach()[:, :, 20 // 3 : 2 * 20 // 3]
            right = test_pred.cpu().detach()[:, :, 2 * 20 // 3:]

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
            
            """qualitative analysis"""
            # region: Qualitative Analysis
            # l, c, r = 0, 0, 0
            # if random.uniform(0, 1) > 0.8:
            #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                
            #     ax[0].imshow(test_images[0].cpu().squeeze(1).squeeze(0).numpy(), cmap="gray")
            #     ax[0].set_title("Input Image")
            #     ax[0].xaxis.set_ticks_position("top")
                
            #     ax[1].imshow(test_ground_truth[0].cpu().squeeze(1).squeeze(0).numpy(), cmap="gray")
            #     ax[1].set_title("Ground Truth")
            #     ax[1].xaxis.set_ticks_position("top")

            #     ax[1].set_title(f"Ground Truth: {'Safe' if labels[0, 0] else 'Unsafe'} | {'Safe' if labels[0, 1] else 'Unsafe'} | {'Safe' if labels[0, 2] else 'Unsafe'}")

                
            #     ax[2].imshow(test_pred[0].cpu().numpy(), cmap="gray")
            #     ax[2].set_title("Predicted Depth")
            #     ax[2].xaxis.set_ticks_position("top")
            #     ax[2].set_title(f"Predicted: {'Safe' if left_prob[0] else 'Unsafe'} | {'Safe' if center_prob[0] else 'Unsafe'} | {'Safe' if right_prob[0] else 'Unsafe'}")
                
            #     for axis in [ax[1], ax[2]]:
            #         axis.axvline(x= 20 // 3, color='r', linestyle='--')
            #         axis.axvline(x=2 * 20 // 3, color='r', linestyle='--')
                
                
            #     a = test_ground_truth[0].cpu().numpy().flatten()
            #     b = test_pred[0].cpu().numpy().flatten()
            #     p = stats.pearsonr(a, b)[0]
            #     r2 = r2_score(a, b)
                
            #     l += 1 if left_prob[0] == labels[0, 0] else 0
            #     r += 1 if right_prob[0] == labels[0, 2] else 0
            #     c += 1 if center_prob[0] == labels[0, 1] else 0
                
            #     fig.suptitle(f"R2: {r2:.3f} | Pearson: {p:.3f}, Acc: {l + c + r} / 3")
                
            #     plt.show()
            # endregion


    # Metrics calculation and logging on wandb
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

    for run in range(1, 3):
        # run is used to select the configuration (train, ultrasound, tethys, train_room, model_name)
        if run == 1:
            train = True
            ultrasound = False
            tethys = True
            train_room = True
            model_name = ""
        else:
            train = True
            ultrasound = True
            tethys = True
            train_room = True
            model_name = ""

        print(f"Run = {run}")
        print(f"train = {train}, ultrasound = {ultrasound}, tethys = {tethys}, train_room = {train_room}")
        seeds_list = [0, 42, 123, 666, 999]
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
                
                df_train = pd.read_pickle("/data/df_train.pkl")
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
                
                df_train = pd.read_pickle("data/df_train.pkl")
                if train_room:
                    df_test = pd.read_pickle("data/df_test.pkl")
                else:
                    df_test = pd.read_pickle("data/df_test_new_room.pkl")

            # model's name setting
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

            # data preprocessing
            input_transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

            ground_truth_transform = transforms.Compose(
                [
                    transforms.Resize((20, 20)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

            # datasets creation
            dataset = CustomDataset(
                data=df_train,
                input_transform=input_transform,
                ground_truth_transform=ground_truth_transform,
                model_name="fcnn",
                train=True,
            )
            
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

            # dataset splitting
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
            
            # model complexity calculation (MACs and number of parameters)
            # fake_input = (1, 1, 160, 160)
            # n_mac, n_params = get_model_complexity_info(model, fake_input, as_strings=True, backend='pytorch', print_per_layer_stat=True)
            # print('FLOPs: ', n_mac)
            # print('Params: ', n_params)
            # exit()


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

                        train_images, ground_truth_images, distance, _ = batch

                        train_images = train_images.to(device)
                        ground_truth_images = ground_truth_images.to(device)
                        distance = distance.to(device)

                        optimizer.zero_grad()

                        if ultrasound:
                            pred = model(train_images, distance).to(device)
                        else:
                            pred = model(train_images).to(device)

                        # pred.shape = (batch, 20, 20)
                        # ground_truth_images.shep = (batch, 1, 20, 20)
                        loss = criterion(pred, ground_truth_images.squeeze(1))

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

                        # validation
                        num_batches = 0

                        for idx, val_batch in enumerate(val_dataloader):

                            val_image, val_ground_truth, distance, _ = val_batch

                            val_image = val_image.to(device)
                            val_ground_truth = val_ground_truth.to(device)
                            distance = distance.to(device)

                            if ultrasound:
                                val_pred = model(val_image, distance).to(device)
                            else:
                                val_pred = model(val_image).to(device)

                            val_pred_stat.extend(val_pred.cpu().numpy().flatten())
                            val_ground_truth_stat.extend(val_ground_truth.cpu().numpy().flatten())

                            loss = criterion(val_pred, val_ground_truth.squeeze(1))
                            
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
                                torch.save(model.state_dict(), f"/data/{model_name}.pth") # tethys
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
                model.load_state_dict(torch.load(f"/data/{model_name}.pth", map_location=device)) # tethys
            else:
                model.load_state_dict(torch.load(f"data/{model_name}.pth", map_location=device)) # local

            # test model function
            test_model(model, test_dataloader, ultrasound, criterion, device, wandb, "_without_smoke")
            test_model(model, test_dataloader_smoke, ultrasound, criterion, device, wandb, "_with_smoke")
            
            wandb.finish()

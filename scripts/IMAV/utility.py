#-----------------------------------------------------------------------------#
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
# File:    utility .py                                                        #
# Author:  Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
#          Vlad Niculescu   <vladn@iis.ee.ethz.ch>                            #
# Date:    01.06.2021                                                         #
#-----------------------------------------------------------------------------#

# essentials
import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

################################################################################
# Dataset Loader
################################################################################

class DronetDatasetV3(Dataset):
    """Dronet dataset."""

    def __init__(self, transform=None, dataset=None, dataset_type='imav', selected_partition=None, labels_preprocessing=False, classification=False, remove_yaw_rate_zero=False, mod_labels=False):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.dataset= dataset
        self.dataset_type = dataset_type
        self.selected_partition = selected_partition
        self.labels_preprocessing = labels_preprocessing
        self.filenames = []
        if self.dataset_type == 'imav': self.labels = [[] for _ in range(7)]
        self.mod_labels=mod_labels

        assert selected_partition in ['train', 'valid', 'test', 'all'], 'Error in the partition name you selected'

        for acquisition in self.dataset.acquisitions:
            if self.dataset_type == 'imav':
                names, partition, edge, target_yaw, l_p_of_collision, c_p_of_collision, r_p_of_collision = self.read_csv_label(acquisition)

            # IMAV DATASET #
            if self.dataset_type == 'imav':
                # filter images by partition
                if selected_partition != 'all':
                    data_filter = (partition == selected_partition)
                    names = names[data_filter]
                    partition = partition[data_filter]
                    edge = edge[data_filter]
                    target_yaw = target_yaw[data_filter]
                    l_p_of_collision = l_p_of_collision[data_filter]
                    c_p_of_collision = c_p_of_collision[data_filter]
                    r_p_of_collision = r_p_of_collision[data_filter]

                if not self.mod_labels:
                    # append labels and filenames to a list
                    self.filenames.extend(names)
                    self.labels[0].extend((edge=='E_label.EDGE').astype(np.int))
                    self.labels[1].extend((edge=='E_label.NO_EDGE').astype(np.int))
                    self.labels[2].extend((edge=='E_label.CORNER').astype(np.int))
                    self.labels[3].extend(target_yaw)
                    self.labels[4].extend(l_p_of_collision)
                    self.labels[5].extend(c_p_of_collision)
                    self.labels[6].extend(r_p_of_collision)

            # check if number of images is matching number of labels:
            if not (len(self.filenames) == len(self.labels[0]) == len(self.labels[1])):
                raise RuntimeError("Mismatch in the number of images and labels: images %d, labels_yaw %d, labels_collision %d" % (len(self.filenames), len(self.labels[0]), len(self.labels[1])))

    def __len__(self):
        # check if number of images is matching number of labels:
        if len(self.filenames) == len(self.labels[0]) == len(self.labels[1]):
            return len(self.filenames)
        else:
            raise RuntimeError("DronetDataset size error: filenames %d, label_yaw %d, label_collision %d", len(self.filenames), len(self.labels[0]), len(self.labels[1]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.filenames[idx], 'rb') as f:
            _image = Image.open(f)
            _image.load()
        
        _label = [  self.labels[0][idx],
                    self.labels[1][idx],
                    self.labels[2][idx],
                    self.labels[3][idx],
                    self.labels[4][idx],
                    self.labels[5][idx],
                    self.labels[6][idx],
                    ]  # [edge, no_edge, corner, yaw_target, left_pcoll, center_pcoll. right_pcoll]
        
        _filename = self.filenames[idx]

        if self.transform:
            _image = self.transform(_image)
            _label = torch.tensor(_label, dtype=torch.float32)

        sample = (_image, _label, _filename)
        return sample

    def normalize_yaw_rate(self, yaw_rate, normalization=90):
        return yaw_rate/normalization

    def read_csv_label(self, acquisition):
        filename = join(acquisition.path, acquisition.PARTITIONED_LABELS_FILENAME)
        df = pd.read_csv(filename)

        try:
            if self.dataset_type == 'imav':
                filenames = (acquisition.images_dir_path + os.sep + df['filename']).to_numpy(dtype=np.str)
                partition = df['partition'].to_numpy(dtype=np.str)
                edge = df['label_edge'].to_numpy(dtype=np.str)
                target_yaw  = df['label_target_yaw'].to_numpy(dtype=np.float)
                l_p_of_collision = df['label_l_p_of_collision'].to_numpy(dtype=np.int)
                c_p_of_collision = df['label_c_p_of_collision'].to_numpy(dtype=np.int)
                r_p_of_collision = df['label_r_p_of_collision'].to_numpy(dtype=np.int)
                return filenames, partition, edge, target_yaw, l_p_of_collision, c_p_of_collision, r_p_of_collision


        except OSError as e:
            print("No labels found in dir", filename)

################################################################################
# Ultrasonic scalar input dataset
################################################################################
class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        input_transform=None,
        train=True,
        smoke=False,
    ):

        self.data = data
        self.input_transform = input_transform
        self.train = train
        self.smoke = smoke

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        distance = self.data.iloc[idx, 4] / 6.4516
        
        if self.train:
            images = self.data.iloc[idx, 9]
        else:
            if self.smoke:
                images = self.data.iloc[idx, 9]
            else:
                images = self.data.iloc[idx, 5]
        
        labels = self.data.iloc[idx, 8]

        images = Image.fromarray(images)

        distance = torch.tensor(distance, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.input_transform:
            images = self.input_transform(images)
            
        return images, labels, distance


################################################################################
# Logging
################################################################################
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt # format. example ':6.2f'
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def write_log(logs_path, log_str, prefix='train', should_print=True, mode='a', end='\n'):
    # Arguments:
    #   - prefix: the name of the log file:  logs_path/'prefix'.log
    with open(join(logs_path, '%s.log' % prefix), mode) as fout:
        fout.write(log_str + end)
        fout.flush()
    if should_print:
        print(log_str)

################################################################################
# Loss Functions and Evaluation Metrics
################################################################################
# steering type=0
def custom_mse(y_true, y_pred, device): # Mean Squared Error
    target_s = y_true[:,0].squeeze()
    input_s  = y_pred[0]
    loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    output_MSE = loss_MSE(input_s, target_s).to(device)
    return output_MSE


# collision type=1
def custom_accuracy(y_true, y_pred, device):

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    acc = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    pre = 0.0
    rec = 0.0
    f1 = 0.0

    pred =  y_pred[1] >= 0.5
    truth = y_true[:,1].squeeze() >= 0.5
    tp += pred.mul(truth).sum(0).float()
    tn += (~pred).mul(~truth).sum(0).float()
    fp += pred.mul(~truth).sum(0).float()
    fn += (~pred).mul(truth).sum(0).float()
    acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    
    return acc

# collision type=1
def custom_bce(y_true, y_pred, device): # Debunking loss functions

    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    target_c = y_true[:,1].squeeze()
    input_c  = y_pred[1]
    loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    output_BCE = loss_BCE(input_c, target_c).to(device)
    return output_BCE



# Labels: steering = 0, collision = 1
def custom_loss_v3(y_true, y_pred, device, partial_training=None):

    # output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)
    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)

    # yaw_rate
    # target_s = y_true[:,0].squeeze()
    # input_s  = y_pred[0]
    # loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # output_MSE = loss_MSE(input_s, target_s).to(device)
    # collision
    target_c = y_true[:,1].squeeze()
    input_c  = y_pred[1]
    loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    output_BCE = loss_BCE(input_c, target_c).to(device)
    # loss formula
    if partial_training == 'classification':
        Loss = output_BCE
    # elif partial_training == 'regression':
    #     Loss = output_MSE
    # else:
        # Loss = output_MSE+output_BCE
    return Loss


# steering type=0
def custom_mse_id_nemo(y_true, y_pred, fc_quantum, device): # Mean Squared Error
    target_s = y_true[:,0].squeeze()
    input_s  = y_pred[0]
    input_s_id = input_s * fc_quantum
    loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    output_MSE = loss_MSE(input_s_id, target_s).to(device)
    return output_MSE


# steering type=0
def custom_mse_id_nemo_v2(y_true, y_pred, t, fc_quantum, device): # Mean Squared Error

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    if (t==0).sum().item() > 0:
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        input_s_id =input_s * fc_quantum
        loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        output_MSE = loss_MSE(input_s_id, target_s).to(device)
        return output_MSE, 1
    else:
        return output_MSE, 0


# collision type=1
def custom_accuracy_yawrate_thresholded(y_true, y_pred, device):
    # Takes in a floating point yaw rate prediction, it transformas it into a
    # classification problem by applying thresholds:
    # left=[-1,-0.1], straight=[-0.1,+0.1], right=[+0.1,+1]
    # then it outputs an accuracy
    yaw_rate_labels_c = regression_as_classification(
                                y_true,
                                low_thresh=-0.1,
                                high_thresh=0.1,
                                low_value=-1.0,
                                high_value=1.0)
    yaw_rate_pred_c = regression_as_classification(
                                y_pred,
                                low_thresh=-0.1,
                                high_thresh=0.1,
                                low_value=-1.0,
                                high_value=1.0)

    # print('float predictions:\n',torch.stack([y_true, y_pred], axis=1))
    # print('int predictions:\n',torch.stack([yaw_rate_labels_c, yaw_rate_pred_c], axis=1))
    acc = (yaw_rate_labels_c==yaw_rate_pred_c).float().mean()
    return acc


################################################################################
# Other Utils
################################################################################
def get_fc_quantum(args, model):
    if args.arch == 'dronet_dory':
        fc_quantum = model.fc.get_output_eps(
            model.resBlock3.add.get_output_eps(
                model.get_eps_at('resBlock3.add', eps_in=1./255)
            )
        )
    elif args.arch == 'dronet_dory_no_residuals':
        fc_quantum = model.fc.get_output_eps(
            model.resBlock3.relu2.get_output_eps(
                model.get_eps_at('resBlock3.relu2', eps_in=1./255)
            )
        )
    if args.arch == 'dronet_imav':
        fc_quantum = model.fc.get_output_eps(
            model.resBlock3.add.get_output_eps(
                model.get_eps_at('resBlock3.add', eps_in=1./255)
            )
        )
    else:
        raise ValueError('Doublecheck the architecture that you are trying to use.')
    return fc_quantum

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except:
            print('warning: no bias defined in layer', m)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



# import cv2
# Colors (B, G, R)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GREEN = (105, 255, 105)
RED = (0, 0, 255)
BLUE = (200, 0, 0)
LIGHT_BLUE = (188, 234, 254)
GREEN = (0, 200, 0)
GRAY = (100, 100, 100)
def create_cv2_image(img_path, yaw_rate_label, collision_label, yaw_rate_output, collision_output):

        # --- LOAD IMAGE ---
        img = cv2.imread(img_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #get image size
        height, width = img.shape[0:2]

        # --- image margin from borders ---
        margin = 10 # pixels
        yaw_bar_size = width - 2*margin
        collision_bar_size = height - 2*margin

        # Normalize yaw and collision over image size
        # ground truth
        collision_label = int(collision_label * collision_bar_size)
        yaw_rate_label = yaw_rate_label*90 #optional
        yaw_rate_label = - int(yaw_rate_label)  #minus sign because axis are inverted on opencv

        # network output
        collision_output = int(collision_output * collision_bar_size)
        yaw_rate_output = yaw_rate_output*90 #optional
        yaw_rate_output = - int(yaw_rate_output)  #minus sign because axis are inverted on opencv

        # --- COLLISION ---
        ### GROUND TRUTH BAR ####
        # light blue: EMPTY COLLISION BIN
        img = cv2.line(img,
                        pt1=(margin, margin),    # start_point
                        pt2=(margin, height-margin),   # end_point
                        color= LIGHT_BLUE,
                        thickness= 5)
        # dark blue: 0 = NO COLLISION, 1 COLLISION
        img = cv2.line(img,
                        pt1 =(margin, height - margin - collision_label),    # start_point
                        pt2 =(margin,height-margin),   # end_point
                        color = BLUE,
                        thickness= 5)

        ### OUTPUT BAR ####
        # light blue: EMPTY COLLISION BIN
        img = cv2.line(img,
                        pt1=(margin*2, margin),    # start_point
                        pt2=(margin*2, height-margin),   # end_point
                        color= LIGHT_BLUE,
                        thickness= 5)
        # dark blue: 0 = NO COLLISION, 1 COLLISION
        img = cv2.line(img,
                        pt1 =(margin*2, height - margin - collision_output),    # start_point
                        pt2 =(margin*2,height-margin),   # end_point
                        color = LIGHT_GREEN,
                        thickness= 5)

        # --- YAW RATE ---  light green
        # half circle
        height, width = img.shape[0:2]
        # Ellipse parameters. NOTE: Make sure all the ellipse parameters are int otherwise it raises "TypeError: ellipse() takes at most 5 arguments (8 given)"
        radius = int(40)
        center = (int(width/2), int(height - margin))
        axes = (radius, radius)
        angle = int(0)
        startAngle = int(180)
        endAngle = int(360)
        thickness = int(1)
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, WHITE, thickness)

        # MOVING DOT GROUND TRUTH
        startAngle = int(270+yaw_rate_label)
        endAngle = int(270+yaw_rate_label+1)
        thickness = int(10)
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, BLUE, thickness)

        # MOVING DOT OUTPUT
        startAngle = int(270+yaw_rate_output)
        endAngle = int(270+yaw_rate_output+1)
        thickness = int(10)
        center = (int(width/2), int(height - margin*2))
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, LIGHT_GREEN, thickness)

        return img


def regression_as_classification(input_tensor, low_thresh, high_thresh, low_value, high_value):
    # if tensor[i]<low_thresh: tensor[i]=low_value
    # elif tensor[i]>high_thresh: tensor[i]=high_value
    # else: tensor[i]=0
    tensor=input_tensor.clone()
    tensor[tensor>high_thresh]=high_value
    tensor[tensor<low_thresh]=low_value
    tensor[torch.logical_and(tensor>low_thresh, tensor<high_thresh)] = 0.0
    return tensor


################################################################################
# V2 Utils
################################################################################
def label_steering_to_yawrate_V2(labels, types):
    if (types==0).sum().item() > 0:
        # clamp in [-1;+1] range
        labels[types==0] = torch.clamp(labels[types==0], min=-1, max=1)
        # scale to Yaw-Rate Range
        labels[types==0] = labels[types==0] * 90 # deg/s
        # change variable name
        yawrate_labels = labels
        return yawrate_labels
    else:
        return labels

def output_steering_to_yawrate(outputs):
    # clamp in [-1;+1] range
    outputs[0] = torch.clamp(outputs[0], min=-1, max=1)
    # scale to Yaw-Rate Range
    outputs[0] = outputs[0] * 90
    return outputs

def label_steering_to_yawrate_V3(labels):
    # scale to Yaw-Rate Range
    labels[:,0] = labels[:,0] * 90 # deg/s
    # change variable name
    yawrate_labels = labels
    return yawrate_labels


# steering type=0
def custom_mse_v2(y_true, y_pred, t, device): # Mean Squared Error

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    if (t==0).sum().item() > 0:
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        output_MSE = loss_MSE(input_s, target_s).to(device)
        return output_MSE, 1
    else:
        return output_MSE, 0


# collision type=1
def custom_accuracy_v2(y_true, y_pred, t, device):

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    acc = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    pre = 0.0
    rec = 0.0
    f1 = 0.0

    if (t==1).sum().item() > 0:
        pred =  y_pred[1][t==1] >= 0.5
        truth = y_true[t==1] >= 0.5
        tp += pred.mul(truth).sum(0).float()
        tn += (~pred).mul(~truth).sum(0).float()
        fp += pred.mul(~truth).sum(0).float()
        fn += (~pred).mul(truth).sum(0).float()
        acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
        # avg_pre = nanmean(pre)
        # avg_rec = nanmean(rec)
        # avg_f1 = nanmean(f1)
        return acc, 1
    else:
        return acc, 0


# collision type=1
def custom_bce_v2(y_true, y_pred, t, device): # Debunking loss functions

    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)

    if (t==1).sum().item() > 0:
        target_c = y_true[t==1]
        input_c  = y_pred[1][t==1]
        loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
        output_BCE = loss_BCE(input_c, target_c).to(device)
        return output_BCE, 1
    else:
        return output_BCE, 0


# Labels: steering = 0, collision = 1
def custom_loss_v2(y_true, y_pred, t, epoch, args, device):  #prime 10 epoche traino solo su steering. dal ep 10 inizio a trainare anche su classification

    output_MSE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)
    output_BCE = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True).to(device)

    if args.hard_mining_train:
        global alpha, beta
        batch_size_training = args.batch_size
        k_mse = np.round(batch_size_training-(batch_size_training-10)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0)))))
        k_entropy = np.round(batch_size_training-(batch_size_training-5)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0)))))
        alpha = 1.0
        beta = np.maximum(0.0, 1.0-np.exp(-1.0/10.0*(epoch-10)))

    n_samples_mse = (t==0).sum().item()
    n_samples_entropy = (t==1).sum().item()

    if n_samples_mse > 0:
        k = int(min(k_mse, n_samples_mse)) # k_mse è quello effettivo: prendiamo i top k. calcolo come minimo tra k' (si aggiorna cin modo dinamico) e il minimo delle label tra steering e collision
        target_s = y_true[t==0]
        input_s  = y_pred[0][t==0]
        if args.hard_mining_train:
            # output_MSE = torch.mul((input_s-target_s),(input_s-target_s)).mean()
            l_mse = torch.mul((input_s-target_s),(input_s-target_s))
            output_MSE_value, output_MSE_idx = torch.topk(l_mse, k)
            output_MSE = output_MSE_value.mean().to(device)
        else:
            loss_MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            output_MSE = loss_MSE(input_s, target_s).to(device)

    if n_samples_entropy > 0:
        k = int(min(k_entropy, n_samples_entropy))
        target_c = y_true[t==1]
        input_c  = y_pred[1][t==1]
        if args.hard_mining_train:
            # output_BCE = F.binary_cross_entropy(input_c, target_c, reduce=False).mean()
            l_bce = F.binary_cross_entropy(input_c, target_c, reduce=False)
            output_BCE_value, output_BCE_idx = torch.topk(l_bce, k)
            output_BCE = output_BCE_value.mean().to(device)
        else:
            loss_BCE = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
            output_BCE = loss_BCE(input_c, target_c).to(device)

    if args.hard_mining_train:
        return output_MSE*alpha+output_BCE*beta
    else:
        return output_MSE+output_BCE
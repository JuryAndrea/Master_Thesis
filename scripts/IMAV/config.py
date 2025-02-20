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
# File:    config.py                                                          #
# Author:  Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
#          Vlad Niculescu   <vladn@iis.ee.ethz.ch>                            #
# Date:    18.02.2021                                                         #
#-----------------------------------------------------------------------------#

# This file gathers all the default values for the following scripts:
# training.py, testing.py, evaluation.py, quantize.py

class cfg:
    pass

# default for all scripts
cfg.data_path= './dataset/'
cfg.logs_dir='./logs/'
# cfg.arch='dronet_dory'
cfg.arch='dronet_imav'
cfg.image_size = '160x160'
cfg.depth_mult=1.0
cfg.testing_dataset='original'
cfg.model_weights='model/dronet_v2_dory.pth'
cfg.gpu='0'
cfg.workers=4

# training.py
cfg.model_name = 'pulp_dronet_v3'
cfg.training_batch_size=32
cfg.epochs=100
cfg.learning_rate = 1e-3
cfg.lr_decay = 1e-5
cfg.checkpoint_path = './checkpoints/'
cfg.hard_mining_train = False
# cfg.early_stopping = False
cfg.early_stopping = True
cfg.patience = 15
cfg.delta = 0
cfg.resume_training = False
cfg.aug='all'

# testing.py
cfg.testing_batch_size=32

# evaluation.py
cfg.testing_dataset_evaluation='validation'
cfg.cherry_picking_path='./checkpoints/pulp_dronet_v3/'

# quantize.py
cfg.nemo_export_path = 'nemo_output/'
cfg.nemo_onnx_name = 'pulp_dronet_id_4dory.onnx'

# testing_v2_and_v3.py
###################### Full Dataset #######################
cfg.data_path_v2 = '/scratch/datasets/pulp-dronet-dataset/'
cfg.model_weights_v2 = 'model/dronet_v2_dory_original_himax.pth'
cfg.data_path_v3 = '/home/lamberti/work/dataset/Dataset-PULP-Dronet-V3_noaug/'
cfg.model_weights_v3 = 'model/pulp_dronet_v3_full_dataset_100.pth'
#######################################################
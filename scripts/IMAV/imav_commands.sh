conda activate pulp-dronet-v3-new

# dataset split
python dataset_partitioning_imav.py --data_path=/home/lamberti/work/dataset/imav-dataset/z_50
python dataset_partitioning_imav.py --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50

# 324x324
python training_imav.py --gpu=0 --model_name=imav_all                                         --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=324x324
python training_imav.py --gpu=1 --model_name=imav_edge        --partial_training=edge         --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=324x324
python training_imav.py --gpu=2 --model_name=imav_yaw         --partial_training=yaw          --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=324x324
python training_imav.py --gpu=3 --model_name=imav_collision   --partial_training=collision    --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=324x324

# 162x162
python training_imav.py --gpu=0 --model_name=imav_all                                         --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=162x162
python training_imav.py --gpu=0 --model_name=imav_all_gain1-3                                 --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=162x162
python training_imav.py --gpu=0 --model_name=imav_all_gain0.7-4                               --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=162x162
python training_imav.py --gpu=1 --model_name=imav_edge        --partial_training=edge         --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=162x162
python training_imav.py --gpu=2 --model_name=imav_yaw         --partial_training=yaw          --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=162x162
python training_imav.py --gpu=3 --model_name=imav_collision   --partial_training=collision    --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/ --image_size=162x162

# ----------- newlabels -----------
# 162x162 -- newlabels
python training_imav.py --gpu=3 --model_name=imav_all_newlabels_gain0.7-4             --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/ --image_size=162x162
# 162x162 -- only collision
python training_imav.py --gpu=3 --model_name=imav_all_newlabels_only_coll_gain0.7-4     --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/ --image_size=162x162 --partial_training=collision

# # ----------- testing -----------
# sim
    # no gain
    python training_imav.py --gpu=2 --test --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/training-3st-dataset-aug/162x162/imav_all_newlabels/imav_all_newlabels_100.pth --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/ --image_size=162x162
    # gain [1,3]
    python training_imav.py --gpu=2 --test --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_gain1-3/checkpoint/imav_all_newlabels_gain1-3_60.pth --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/ --image_size=162x162
    # gain [0.7,4]
    python training_imav.py --gpu=2 --test --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_gain0.7-4/checkpoint/imav_all_newlabels_gain0.7-4_25.pth --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/ --image_size=162x162

    # gain [1,3] -- onlycoll
    python training_imav.py --gpu=2 --test --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_only_coll_gain1-3/checkpoint/imav_all_newlabels_only_coll_gain1-3_60.pth --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/ --image_size=162x162


# real
    # ----- old labels -----
    # old labels, no aug
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/training-3st-dataset-aug/162x162/imav_all/imav_all_100.pth
    # old labels,  aug [1,3]
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_gain1-3/checkpoint/imav_all_gain1-3_46.pth
    # old labels,  aug [0.7,4]

    # ----- new labels -----
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_gain0.7-4/checkpoint/imav_all_gain0.7-4_42.pth
    # no gain
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/training-3st-dataset-aug/162x162/imav_all_newlabels/imav_all_newlabels_100.pth
    # gain [1,3]
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_gain1-3/checkpoint/imav_all_newlabels_gain1-3_60.pth
   # gain [1,3] -- onlycoll
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_only_coll_gain1-3/checkpoint/imav_all_newlabels_only_coll_gain1-3_60.pth
    # gain [0.7,4]
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_gain0.7-4/checkpoint/imav_all_newlabels_gain0.7-4_25.pth
    # gain [0.7,4] -- onlycoll
    python training_imav.py --gpu=2 --test --true_data  --model_name=debug   --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-real/dataset_LB_2_labeled_5interval --image_size=162x162 --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_newlabels_only_coll_gain0.7-4/checkpoint/imav_all_newlabels_only_coll_gain0.7-4_13.pth



# ----------- Quantize -----------
# 162x162 images, no gain aug, old labels (edges are not obstacles)
python quantize_imav.py --gpu=1  --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/    --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/training-3st-dataset-aug/162x162/imav_all_newlabels/imav_all_newlabels_100.pth --image_size=162x162 --export_path=/home/lamberti/work/IMAV2022/pulp-dronet-v3/nemo-dory/nemo_output_aug_newlabels_162x162
# 162x162 images, no gain aug, new labels (edges are  obstacles)
python quantize_imav.py --gpu=1  --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset-newlabels/z_50/    --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/training-3st-dataset-aug/162x162/imav_all_newlabels/imav_all_newlabels_100.pth --image_size=162x162 --export_path=/home/lamberti/work/IMAV2022/pulp-dronet-v3/nemo-dory/nemo_output_aug_newlabels_162x162


# 162x162 images, standard labels (edges are not obstacles)
python quantize_imav.py --gpu=0  --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/              --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_gain1-3/checkpoint/imav_all_gain1-3_48.pth        --image_size=162x162 --export_path=/home/lamberti/work/IMAV2022/pulp-dronet-v3/nemo-dory/nemo_output_aug_162x162_OLDLABELS1-3
python quantize_imav.py --gpu=0  --block_type=ResBlock   --depth_mult=1.0    --arch=dronet_imav --data_path=/home/lamberti/work/dataset/imav-dataset/z_50/              --model_weights=/home/lamberti/work/IMAV2022/pulp-dronet-v3/training/imav_all_gain0.7-4/checkpoint/imav_all_gain0.7-4_42.pth    --image_size=162x162 --export_path=/home/lamberti/work/IMAV2022/pulp-dronet-v3/nemo-dory/nemo_output_aug_162x162_OLDLABELS0.7-4


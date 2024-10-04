import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Class used to create a custom dataset
class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        input_transform=None,
        ground_truth_transform=None,
        model_name=None,
        train=True,
        smoke=False,
    ):

        self.data = data
        self.input_transform = input_transform
        self.ground_truth_transform = ground_truth_transform
        self.model_name = model_name
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

        ground_truth_images = self.data.iloc[idx, 6]
        
        if self.model_name == "unet":
            labels = self.data.iloc[idx, 8]
        else:
            labels = self.data.iloc[idx, 7]

        images = Image.fromarray(images)
        ground_truth_images = Image.fromarray(ground_truth_images)

        distance = torch.tensor(distance, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.input_transform and self.ground_truth_transform:
            images = self.input_transform(images)
            ground_truth_images = self.ground_truth_transform(ground_truth_images)

        return images, ground_truth_images, distance, labels


# Function used to perform thresholding on the portion of the image within 2 meters
# if at least 10% of the pixels are within 2 meters, the portion of the image is considered unsafe (0) otherwise safe (1)
def perform_thresholding(image):

    # 1 : 6.4516 = x : 2
    x = 2 * 1 / 6.4516

    # pixels within 2 meters
    image = np.array(image)
    pixels_within_threshold = np.sum(image <= x, axis=(1, 2))

    total_pixels = np.prod(image.shape[1:])

    # percentage of pixels within 2 meters
    percentage = pixels_within_threshold / total_pixels * 100
    
    # 0 = unsafe, 1 = safe
    return (percentage < 10).astype(int)
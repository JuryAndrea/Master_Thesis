import torch
import random
from torch import nn

# Python file containing the training models

"""original MkModel"""
class MkModel(nn.Module):
    def __init__(self):
        super(MkModel, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu7 = nn.ReLU()


    def forward(self, x):
        
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        
        # block A
        x = self.relu5(self.bn5(self.sep_conv4(x)))
        
        # block A
        x = self.relu6(self.bn6(self.conv2(x)))
        
        # block A
        x = self.relu7(self.bn7(self.conv3(x)))

        return x[:, 1, :, :]

"""MkModel with ultrasound"""
class MkModel_v2(nn.Module):
    def __init__(self):
        super(MkModel_v2, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=3, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()
        
        # Fusion MLP
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.relu7 = nn.ReLU()


    def forward(self, x, y):
        
        # x is the image, y is the ultrasound scalar value
        
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 1, 1, 1)
        y = y.expand(y.size(0), 1, 160, 160)
        x = torch.cat((x, y), 1)
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        # block A
        x = self.relu8(self.bn5(self.sep_conv4(x)))
        
        # block A
        x = self.relu9(self.bn6(self.conv2(x)))
        
        # block A
        x = self.relu10(self.bn7(self.conv3(x)))
        
        return x[:, 1, :, :]

class MkModel_v3(nn.Module):
    def __init__(self):
        super(MkModel_v3, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=5, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()
        
        # Fusion MLP
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.relu7 = nn.ReLU()


    def forward(self, x, y):
        
        # x is the image, y is the ultrasound scalar value
        
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 1, 1, 1)
        y = y.expand(y.size(0), 1, 80, 80)
        x = torch.cat((x, y), 1)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        # block A
        x = self.relu8(self.bn5(self.sep_conv4(x)))
        
        # block A
        x = self.relu9(self.bn6(self.conv2(x)))
        
        # block A
        x = self.relu10(self.bn7(self.conv3(x)))
        
        return x[:, 1, :, :]

class MkModel_v4(nn.Module):
    def __init__(self):
        super(MkModel_v4, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=12, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()
        
        # Fusion MLP
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.relu7 = nn.ReLU()


    def forward(self, x, y):
        
        # x is the image, y is the ultrasound scalar value
        
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 4, 1, 1)
        y = y.expand(y.size(0), 4, 40, 40)
        x = torch.cat((x, y), 1)
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        # block A
        x = self.relu8(self.bn5(self.sep_conv4(x)))
        
        # block A
        x = self.relu9(self.bn6(self.conv2(x)))
        
        # block A
        x = self.relu10(self.bn7(self.conv3(x)))
        
        return x[:, 1, :, :]

class MkModel_v5(nn.Module):
    def __init__(self):
        super(MkModel_v5, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()
        
        # Fusion MLP
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.relu7 = nn.ReLU()


    def forward(self, x, y):
        
        # x is the image, y is the ultrasound scalar value
        # breakpoint()
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 4, 1, 1)
        y = y.expand(y.size(0), 4, 20, 20)
        x = torch.cat((x, y), 1)
        
        # block A
        x = self.relu8(self.bn5(self.sep_conv4(x)))
        
        # block A
        x = self.relu9(self.bn6(self.conv2(x)))
        
        # block A
        x = self.relu10(self.bn7(self.conv3(x)))
        
        return x[:, 1, :, :]

class MkModel_v6(nn.Module):
    def __init__(self):
        super(MkModel_v6, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=36, out_channels=36, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(36)
        self.relu9 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()
        
        # Fusion MLP
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.relu7 = nn.ReLU()


    def forward(self, x, y):
        
        # x is the image, y is the ultrasound scalar value
        
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        # block A
        x = self.relu8(self.bn5(self.sep_conv4(x)))
        
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 4, 1, 1)
        y = y.expand(y.size(0), 4, 20, 20)
        x = torch.cat((x, y), 1)
        
        # block A
        x = self.relu9(self.bn6(self.conv2(x)))
        
        # block A
        x = self.relu10(self.bn7(self.conv3(x)))
        
        return x[:, 1, :, :]

class MkModel_v7(nn.Module):
    def __init__(self):
        super(MkModel_v7, self).__init__()

        # Block A
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

        # Block B
        self.sep_conv1 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Block B
        self.sep_conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block A
        self.sep_conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        # Block A
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=1, padding=0
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        # Block A
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=3, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()
        
        # Fusion MLP
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.relu7 = nn.ReLU()


    def forward(self, x, y):
        
        # x is the image, y is the ultrasound scalar value
        # block A
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # block B
        x = self.relu2(self.bn2(self.sep_conv1(x)))
        x = self.pool1(x)
        
        # block B
        x = self.relu3(self.bn3(self.sep_conv2(x)))
        x = self.pool2(x)
        
        
        # block B
        x = self.relu4(self.bn4(self.sep_conv3(x)))
        x = self.pool3(x)
        
        # block A
        x = self.relu8(self.bn5(self.sep_conv4(x)))
        
        # block A
        x = self.relu9(self.bn6(self.conv2(x)))
        
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 4, 1, 1)
        y = y.expand(y.size(0), 4, 20, 20)
        x = torch.cat((x, y), 1)
        
        # block A
        x = self.relu10(self.bn7(self.conv3(x)))
        
        return x[:, 1, :, :]


#### --------------------------------------------------------------------------------------------- ###
#### --------------------------------------------------------------------------------------------- ###


"""U-Net utilities"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)




"""My U-Net"""
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_convolution_1 = DownSample(1, 2)
        self.down_convolution_2 = DownSample(2, 4)
        self.down_convolution_3 = DownSample(4, 8)

        self.bottle_neck = DoubleConv(8, 16)

        self.up_convolution_1 = UpSample(16, 8)
        self.up_convolution_2 = UpSample(8, 4)
        self.up_convolution_3 = UpSample(4, 2)

        self.out = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x):
        # 160 x 160
        down_1, p1 = self.down_convolution_1(x)
        # 80 x 80
        
        # 80 x 80
        down_2, p2 = self.down_convolution_2(p1)
        # 40 x 40
        
        # 40 x 40
        down_3, p3 = self.down_convolution_3(p2)
        # 20 x 20
        
        # 20 x 20
        b = self.bottle_neck(p3)
        
        # 20 x 20
        up_1 = self.up_convolution_1(b, down_3)
        # 40 x 40
        up_2 = self.up_convolution_2(up_1, down_2)
        # 80 x 80
        up_3 = self.up_convolution_3(up_2, down_1)
        # 160 x 160
        out = self.out(up_3)

        return out


"""My U-Net with ultrasound"""
class UNet_v2(nn.Module):
    def __init__(self):
        super(UNet_v2, self).__init__()

        self.down_convolution_1 = DownSample(1, 2)
        self.down_convolution_2 = DownSample(2, 4)
        self.down_convolution_3 = DownSample(4, 8)

        self.bottle_neck = DoubleConv(12, 16)

        self.up_convolution_1 = UpSample(16, 8)
        self.up_convolution_2 = UpSample(8, 4)
        self.up_convolution_3 = UpSample(4, 2)

        self.out = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        
        self.fc1 = nn.Linear(in_features=1, out_features=64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.relu7 = nn.ReLU()

    def forward(self, x, y):
        
        # x is the image, y is the ultraound value
        breakpoint()
        # 160 x 160
        down_1, p1 = self.down_convolution_1(x)
        # 80 x 80
        
        # 80 x 80
        down_2, p2 = self.down_convolution_2(p1)
        # 40 x 40
        
        # 40 x 40
        down_3, p3 = self.down_convolution_3(p2)
        # 20 x 20
        
        # FUSION MLP
        y = y.view(y.size(0), 1)
        y = self.relu5(self.fc1(y))
        y = self.relu6(self.fc2(y))
        y = self.relu7(self.fc3(y))
        y = y.view(y.size(0), 4, 1, 1)
        y = y.expand(y.size(0), 4, 20, 20)
        
        p3 = torch.cat((p3, y), 1)
        
        # 20 x 20
        b = self.bottle_neck(p3)

        # 20 x 20
        up_1 = self.up_convolution_1(b, down_3)
        # 40 x 40
        up_2 = self.up_convolution_2(up_1, down_2)
        # 80 x 80
        up_3 = self.up_convolution_3(up_2, down_1)
        # 160 x 160
        
        out = self.out(up_3)

        return out


#### --------------------------------------------------------------------------------------------- ###
#### --------------------------------------------------------------------------------------------- ###


"""U-Net from the original paper"""
class UNet_original(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_convolution_1 = DownSample(1, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)
        
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)
        
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)

        return out



"""U-Net from the original paper with ultrasound"""
class UNet_original_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_convolution_1 = DownSample(1, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(257, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)
        
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x, scalar_value):

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        
        scalar_value = scalar_value / 6.4516
        y = torch.stack([torch.full((1, 40, 40), val.item()) for val in scalar_value])
        y = y.to(x.device)
        
        p3 = torch.cat([p3, y], 1)
        
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)
        
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)

        return out
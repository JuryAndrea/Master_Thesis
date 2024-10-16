# Master Thesis Repository

This repository contains the code and resources for the master thesis project of Jury Andrea D'Onofrio. It includes various scripts, definitions, and Jupyter notebooks used for data analysis, plotting, and simulations.

## Repository Structure

- **definitions/**: Contains definition files used for singleton in tethys.
    - **definition.def**: Contains the definition of the singleton creatable with the command `singularity build --fakeroot ml_container.sif definition.def`
&nbsp;
- **jupyter_notebook/**: Includes a Jupyter notebook for data visualization and analysis.
    - **data/**: Contains the metrics values of the models during the training and testing phases in CSV format downloaded from wandb.
    - **plot.ipynb**: A Jupyter notebook for data visualization and analysis.
&nbsp;
- **scripts/**: Contains Python scripts for training and testing the models. Moreover, it contains the `pth` files of the trained models.
    - **data/**: Contains the data used for training and testing the models.
    - **models/**: Contains the final trained models in `pth` format and several models used for validate the results with different seeds.
    - **IMAV/**: Contains all the files for the IMAV model. This model is a binary classifier that predicts the presence of an obstacle in each third of the image captured by the camera (left, center, right). The most important files are:
        - **training_imav.py**: A script that defines the training and testing for the IMAV model.
        run `python training_imav.py`.
        - **utility.py**: A script containing utility functions for creating the dataset.
        - **models/dronet_imav.py**: A script containing the architectures of the IMAV model.
    - **FCNN.py**: A script that defines the training and testing for the Fully Convolutional Neural Network model. This model is a depth estimation model that predicts the depth of the image captured by the camera. Then the depth is used to predict the presence of an obstacle in the image as a binary classifier.
    run `python FCNN.py`.
    - **training_models.py**: A script containing each model's architecture.
    - **trivial.py**: A script containing the testing for the trivial model.
    run `python trivial.py`.
    - **unet.py**: A script containing the training and testing for customized version of the U-Net model.
    This model is a depth estimation model that predicts the depth of the image captured by the camera. Then the depth is used to predict the presence of an obstacle in the image as a binary classifier.
    run `python unet.py`.
    - **unet_teacher.py**: NOT USED. It contains the training and testing for original version of the U-Net model.
    - **utilities.py**: A script containing utility functions for creating the dataset and the dataloader and transforming depth estimations' models into binary classifiers.
&nbsp;
- **Webots.zip**: A zipped folder with Webots simulation files, including the worlds, the controllers, and the proto files. The controllers are written in Python and they start automatically when the simulation is launched. However, before running the simulation, check if the controller is set into the `controller` field of the CrazyflieCloud node in the world object tree.

Note that the scripts are designed to be run on tethys, especially for the training phase (be sure to create a screen for each python file). The testing phase can be run on local machines with the required modifications.

## Installation

1. Clone this repository:
    `git clone https://github.com/JuryAndrea/Final_Master_Thesis.git`
<br>
2. Install the required dependencies after navigating to the desired directory:
    `pip install -r requirements.txt`



# Importing necessary libraries
import numpy as np
import torch
import torch.utils.data as Data
from torch import Tensor
from MLPipeline.CNNNet import CNNNet 
from MLPipeline.Train import TrainModel  
from MLPipeline.CreateDataset import CreateDataset  

# Define the root directory for your data
ROOT_DIR = "Input/"

# Define the training and testing data folders
Training_folder = ROOT_DIR + "Data/Training_data"
Test_folder = ROOT_DIR + "Data/Testing_Data"

# Define the image dimensions
IMG_WIDTH = 200  # Image width
IMG_HEIGHT = 200  # Image height

# Load training data
print("Loading Training Data")
Train_img_data, train_class_name = CreateDataset().create_dataset(Train_folder, IMG_WIDTH, IMG_HEIGHT)
print("Training Data Loaded")

# Load testing data
print("Loading Testing Data")
Test_img_data, test_class_name = CreateDataset().create_dataset(Test_folder, IMG_WIDTH, IMG_HEIGHT)
print("Testing Data Loaded")

# Create PyTorch datasets from the loaded data
torch_dataset_train = Data.TensorDataset(Tensor(np.array(Train_img_data)), Tensor(np.array(train_class_name))
torch_dataset_test = Data.TensorDataset(Tensor(np.array(Test_img_data)), Tensor(np.array(test_class_name))

# Define data loaders for training and testing
trainloader = torch.utils.data.DataLoader(torch_dataset_train, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=8, shuffle=True)

# Create the model
model = CNNNet()

# Train the model using the training data
TrainModel(model, ROOT_DIR, trainloader, testloader)

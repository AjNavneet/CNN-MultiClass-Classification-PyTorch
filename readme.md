# Multiclass Image Classification using CNN with PyTorch

## Overview
Convolutional Neural Network (CNN) is a deep learning algorithm that learns directly from data, eliminating the need for manual feature extraction. CNNs are particularly useful for image data, helping recognize patterns in images. In this project, we build a CNN model for image classification, categorizing images into classes such as social security cards, driving licenses, and others. We have used PyTorch for building the model, which offers dynamic computational graphs and a Pythonic interface.

---

## Aim

To build a Convolutional Neural Network model to classify images into different classes in PyTorch

---

## Tech Stack
- Language: `Python`
- Libraries: `PyTorch`, `pandas`, `matplotlib`, `NumPy`, `opencv_python_headless`, `torchvision`

---

## Data Description
The dataset includes images of driving licenses, social security cards, and other categories. These images are of varying shapes and sizes, which are preprocessed before modeling.

---

## Approach
1. Data Loading
2. Data Preprocessing
   - Resizing and scaling of the images
   - Encoding of the class labels
3. Model Building and Training
   - CNN model building in PyTorch

---

## Modular Code Overview
After unzipping the `pytorch_cnn.zip` file, you'll find the following folders:

1. Input: Contains training and testing data for image classification.
2. Notebook: Contains the Jupyter notebook file for the project.
3. ML_Pipeline: A folder with Python functions organized in different files. These functions are called inside `Engine.py`.
4. Output: Contains the saved CNN model.
5. requirements.txt: Lists all the required libraries with respective versions. Install them using `pip install -r requirements.txt`.
6. `Readme.md`: This file with instructions for running the code.

---

## Takeaways

1. Architecture of CNN
2. Extracting features from images
3. kernels in CNN
4. Padding and pooling
5. Data loading in PyTorch
6. Data preprocessing in PyTorch
7. CNN implementation in PyTorch

---





---

# Pytorch

- PyTorch is an open source machine learning library for Python and is completely based on Torch. 
- It is primarily used for applications such as natural language processing
- PyTorch redesigns and implements Torch in Python while sharing the same core C libraries for the backend code.
- PyTorch developers tuned this back-end code to run Python efficiently. They also kept the GPU based hardware acceleration as well as the extensibility features that made Lua-based Torch.

---

## Getting Started

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `CNN.ipynb`

---

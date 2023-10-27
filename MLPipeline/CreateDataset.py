import numpy as np
import os
import cv2

class CreateDataset:

    # Creating the image data and the labels from the images
    def create_dataset(self, Train_folder, IMG_WIDTH, IMG_HEIGHT):
        img_data_array = []  # List to store image data
        class_name = []      # List to store class labels

        # Define a dictionary that maps class names to one-hot encoded labels
        classes = {'driving_license': [1, 0, 0], 'others': [0, 1, 0], 'social_security': [0, 0, 1]}

        # Loop through the subfolders in the 'Train_folder'
        for PATH in os.listdir(Train_folder):
            # Skip the .DS_Store folder (common on macOS)
            if PATH == ".DS_Store":
                continue

            for file in os.listdir(os.path.join(Train_folder, PATH)):
                image_path = os.path.join(Train_folder, PATH, file)

                # Read the image in RGB format and resize it
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float64')
                # You can normalize the pixel values by dividing by 255 if needed
                # image /= 255

                if len(image.shape) == 3:
                    img_data_array.append(np.array(image).reshape([3, IMG_HEIGHT, IMG_WIDTH]))
                    class_name.append(classes[PATH])

        return img_data_array, class_name

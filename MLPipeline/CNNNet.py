import torch.nn as nn

# Define the model architecture
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        # Define the CNN layers
        self.cnn_layers = nn.Sequential(
            # Convolutional layer 1: 16 filters, 5x5 kernel, 2x2 stride, 2x2 padding
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(inplace=True),
            # MaxPooling layer 1: 2x2 kernel, 2x2 stride
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolutional layer 2: 3 filters, 50x50 kernel, 1x1 stride
            nn.Conv2d(16, 3, kernel_size=(50, 50), stride=(1, 1)),
            # MaxPooling layer 2: 1x1 kernel, 1x1 stride
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)
        )

        # Define the linear (fully connected) layers
        self.linear_layers = nn.Sequential(
            # Linear layer: input size 3, output size 3
            nn.Linear(3, 3)
        )

    # Define the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = self.linear_layers(x)
        return x

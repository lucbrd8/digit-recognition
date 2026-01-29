'''
detection_model.py - Contains the two models for digit recognition

'''

import torch.nn as nn
from torchsummary import summary

class BaseModel(nn.Module):

    def __init__(self,img_size = 28*28):
        super().__init__()

        self.input_conv_layer = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.mid_conv_layer = nn.Conv2d(
            in_channels = 16,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.output_layer = nn.Linear(
            in_features = 16*img_size//16,
            out_features = 10
        )
        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        self.input_norm = nn.BatchNorm2d(num_features = 16)
        self.mid_norm = nn.BatchNorm2d(num_features=16)
        self.vectorize = nn.Flatten()
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.pool(self.relu(self.input_norm(self.input_conv_layer(x))))
        x = self.pool(self.relu(self.mid_norm(self.mid_conv_layer(x))))
        return self.output_layer(self.vectorize(x))
    
if __name__ == "__main__":
    model = BaseModel()
    summary(model, input_size=(1,28,28))
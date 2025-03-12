from torchvision.models import shufflenet_v2_x0_5
from torchvision.models.shufflenetv2 import ShuffleNet_V2_X0_5_Weights
import torch.nn as nn
from config import configs

def LightCDC(num_classes=2, device='cuda'):

    # Loading ShuffleNetV2 Model's Pretrained Weights
    shuffleNetV2 = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)

    # Freezing Layers of ShuffleNetV2 Model
    for param in shuffleNetV2.parameters():
        param.requires_grad = True

    # Unfreeze the last classifier layer
    for param in shuffleNetV2.fc.parameters():
        param.requires_grad = True

    # Modifying ShuffleNetV2 Model's Classifier Layer
    in_features = shuffleNetV2.fc.in_features
    num_hidden = 512  # Example of a new hidden layer size

    # Add more layers to the classifier
    shuffleNetV2.fc = nn.Sequential(
        nn.Linear(in_features, num_hidden),
        nn.ReLU(),            # Activation function
        nn.Dropout(0.50),     # Dropout layer
        nn.Linear(num_hidden, num_hidden),  # Corrected to match output of the previous layer
        nn.ReLU(),
        nn.BatchNorm1d(num_hidden),  # Batch normalization
        nn.Linear(num_hidden, configs.output_shape)  # Adjusted for correct number of output classes
    )

    return shuffleNetV2.to(device)
"""
data.py - Script to handle train and test dataloaders
"""


import detection_scripts.config as config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
device = config.device

transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotation aléatoire
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translation aléatoire
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

### Importing MNIST Dataset

train_dataset = datasets.MNIST(config.path,download=True, train = True, transform= transform)
test_dataset = datasets.MNIST(config.path, download=True, train = False, transform = transform)

### Creating train and test dataloader

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)

### Test functions

def show_random_images():
    fig, axes = plt.subplots(2,5,figsize=(12,5))
    index = np.random.randint(0,len(train_dataset),10)
    for i_ax,i_img in enumerate(index) :
        image,label = train_dataset[i_img]
        print(image)
        image = image.squeeze().numpy()
        axes[i_ax//5,i_ax%5].imshow(image, cmap='gray')
        axes[i_ax//5,i_ax%5].set_title(f"Label : {label}")
        plt.axis('off')
    plt.show()

if __name__=="__main__":
    show_random_images()
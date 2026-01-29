"""
training.py - Contains the training loop for the model
"""

import torch
import torch.nn as nn
from detection_model import BaseModel
import config
from data import train_loader, test_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = config.device
base_save_path = "detection_scripts/model_parameters/best_model_base.pth"

model = BaseModel()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)


def train_one_step(model : nn.Module = model,
                   train_data : torch.utils.data.DataLoader = train_loader, 
                   criterion = criterion,  
                   optimizer = optimizer,
                   epoch : int = 0
                   ) -> tuple[dict,float,float]:
    """
    Training function for a single epoch
    Returns : model_settings, training loss
    """
    ### Initializing training and evaluation metrics
    model.train()
    epoch_loss,epoch_accuracy = 0,0
    correct,total = 0,0
    train_loop = tqdm(train_loader, desc= f"Training of the epoch {epoch+1}")

    for images,labels in train_loop:

        ### Loading data and calculating outputs
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted_labels = torch.max(outputs.data, 1)

        ### Training the model
        optimizer.zero_grad()
        loss = criterion (outputs,labels)
        loss.backward()
        optimizer.step()

        ### Calculating evaluation metrics
        epoch_loss += loss.item()
        correct += (predicted_labels==labels).sum().item()
        total+=labels.size(0)
        train_loop.set_postfix(loss=loss.item())

    epoch_accuracy = 100* correct/total
    epoch_loss/=len(train_data.dataset)
    return model.state_dict(),epoch_loss,epoch_accuracy


def eval_model(model : nn.Module = model,
               test_data : torch.utils.data.DataLoader = test_loader,
               criterion = criterion
               ) -> tuple[float,float] :
    """
    Evaluation function for a model - calculate the average Cross Entropy Loss for each image
    """
    ### Initializing the evaluation and metrics
    model.eval()
    test_loss, test_accuracy = 0,0
    correct,total = 0,0
    with torch.no_grad():
        for images, labels in test_data:

            ### Loading data and calculating outputs
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted_labels = torch.max(outputs.data,1)

            ### Calculating performance metrics
            loss = criterion (outputs,labels)
            test_loss += loss.item()
            correct += (predicted_labels==labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100*correct/total
    test_loss /= len(test_data.dataset)
    return test_loss,test_accuracy


def train_model(model : nn.Module = model,
                num_epochs : int = config.num_epochs,
                train_data : torch.utils.data.DataLoader = train_loader,
                test_data : torch.utils.data.DataLoader = test_loader,
                criterion = criterion,
                optimizer = optimizer,
                save_path : str = base_save_path
                ) -> tuple[list[float],list[float],dict,float]:
    
    ###Initializing performance monitoring over epochs
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []
    best_model_state = None
    best_loss = float("inf")

    for epoch in range(num_epochs):
        ### Running train and evaluation for the epoch
        model_state,train_loss,train_accuracy_epoch = train_one_step(model,train_data,criterion,optimizer,epoch=epoch)
        test_loss,test_accuracy_epoch = eval_model(model, test_data,criterion)

        ### Verifying improvement
        if test_loss<best_loss:
            best_model_state = model_state
            best_loss = test_loss

        ### Adding metrics to monitoring lists
        train_losses.append(train_loss)
        train_accuracy.append(train_accuracy_epoch)
        test_losses.append(test_loss)
        test_accuracy.append(test_accuracy_epoch)

        ### Displaying information
        print(f"Epoch {epoch+1} finished. Training loss : {train_loss:.3g} | Training accuracy : {train_accuracy_epoch:.3g}| Test loss : {test_loss:.3g}| Test accuracy : {test_accuracy_epoch:.3g}")

    torch.save(best_model_state, save_path)
    return train_losses,test_losses,train_accuracy,test_accuracy,best_loss

def plot_losses(train_losses : list[float],
                test_losses : list[float]) -> None:
    num_epochs = len(train_losses)
    epoch = range(1,num_epochs+1)
    plt.plot(epoch,train_losses, color = "blue", label = "Train")
    plt.plot(epoch,test_losses, color = "red", label = "Test")
    plt.legend(loc = "upper right")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_accuracy(train_accuracy : list[float],
                  test_accuracy : list[float]
                  )-> None:
    num_epochs = len(train_accuracy)
    epoch = range(1,num_epochs +1)
    plt.plot(epoch,train_accuracy, color = "blue", label = "Train")
    plt.plot(epoch,test_accuracy, color = "red", label = "Test")
    plt.legend(loc = "upper right")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

### Running training

if __name__ == "__main__":
    train_losses,test_losses,train_accuracy,test_accuracy,best_loss = train_model()
    plot_losses(train_losses,test_losses)
    plot_accuracy(train_accuracy,test_accuracy)

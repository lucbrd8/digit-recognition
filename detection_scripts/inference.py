import config,detection_model,data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device=config.device

def predict_image(image: torch.tensor,model : nn.Module)->list[float]:
    """
    From an image, this function returns a list of length 10 where list[i] contains the probability that the image is digit i
    """
    model,image=model.to(device),image.to(device)
    logits = model(image)
    probs = nn.functional.softmax(logits,dim=1).detach()
    return probs

def test_script():
    """
    A function to ensure the script written works well
    """
    model = detection_model.BaseModel()
    params_path = "detection_scripts/model_parameters/best_model_base.pth"
    model.load_state_dict(torch.load(params_path,weights_only=True))
    indexes = np.random.randint(0,len(data.train_dataset),10)
    fig, axes = plt.subplots(2,5,figsize=(12,5))
    for i_ax,i_img in enumerate(indexes):
        image,label = data.train_dataset[i_img]
        prediction=predict_image(image.unsqueeze(0),model)
        print(prediction)
        image_show = image.squeeze().numpy()
        axes[i_ax//5,i_ax%5].imshow(image_show, cmap='gray')
        axes[i_ax//5,i_ax%5].set_title(f"Label : {label}")
        plt.axis('off')
    plt.show()


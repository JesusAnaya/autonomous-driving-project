# predict_steering.py

import cv2
import numpy as np
import torch
from model import NvidiaModel
from torchvision import transforms
from config import config

# Load the model and set it to eval mode
model_class = NvidiaModel
model = model_class()
model.load_state_dict(torch.load("../save/model.pt", map_location=torch.device(config.device)))
model.to(config.device)
model.eval()

mean = [0.3568, 0.3770, 0.3691]
std = [0.2121, 0.2040, 0.1968]

def crop_down(image):
    h = image.shape[0]
    w = image.shape[1]
    top = 115
    crop_height = h - top
    return image[top:top + crop_height, :]


def predict_steering_angle(image):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.resize, antialias=True),
        transforms.Normalize(mean, std)
    ])

    image_cropped = crop_down(image)
    frame = transform_img(image_cropped).to(config.device)
    batch_t = torch.unsqueeze(frame, 0)

    # Predictions
    with torch.no_grad():
        y_predict = model(batch_t)

    # Converting prediction to degrees
    pred_degrees = np.degrees(y_predict.item())

    return pred_degrees

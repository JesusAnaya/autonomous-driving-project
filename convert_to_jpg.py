import cv2
import glob
import os
from tqdm import tqdm
import numpy as np

# Input and output folders
input_folder =  "/home/anaya/Devel/tfm_project/datasets/udacity_sim_data_2/IMG"
output_folder = "/home/anaya/Devel/tfm_project/datasets/udacity_sim_data_2/images3"

# If output_folder doesn't exist, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all .png images in the input folder
input_images = glob.glob(os.path.join(input_folder, "*.jpg"))


def add_random_shadow_rgb(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a random shadow intensity and region
    intensity = np.random.uniform(0.5, 0.8)
    x1, x2 = np.random.randint(0, img.shape[1], size=2)

    if x1 > x2:
        x1, x2 = x2, x1

    # Apply the shadow
    hsv[:, x1:x2, 2] = hsv[:, x1:x2, 2] * intensity

    # Convert the image back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_random_brightness_rgb(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate a random brightness offset
    offset = np.random.uniform(-50, 50)

    # Add the offset to the V channel
    hsv[:,:,2] = np.clip(hsv[:,:,2] + offset, 0, 255)

    # Convert the image back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


import torch

def add_random_shadow_bgr_cuda(img):
    hsv = torch.from_numpy(cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2HSV)).cuda()

    intensity = torch.cuda.FloatTensor(1).uniform_(0.5, 0.8)
    x1, x2 = torch.randint(0, img.shape[1], (2,)).cuda()

    if x1 > x2:
        x1, x2 = x2, x1

    hsv[:, x1:x2, 2] = hsv[:, x1:x2, 2] * intensity.item()

    return cv2.cvtColor(hsv.cpu().numpy(), cv2.COLOR_HSV2BGR)

def add_random_brightness_bgr_cuda(img):
    hsv = torch.from_numpy(cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2HSV)).cuda()

    offset = torch.cuda.FloatTensor(1).uniform_(-50, 50)

    hsv[:, :, 2] = torch.clamp(hsv[:, :, 2] + offset.item(), 0, 255)

    return cv2.cvtColor(hsv.cpu().numpy(), cv2.COLOR_HSV2BGR)


counter = 0

# Iterate over all input images
for img_path in tqdm(input_images, desc="Converting images", unit="image"):
    # Read the image in color (default setting)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # resize to w:320, h:180
    img = cv2.resize(img, (320, 160))

    # crop the top 20 pixels to convert from 320x180 to 320x160

    img = img[44:-24, 20:-20, :]

    img = add_random_shadow_bgr_cuda(img)

    #img = img[60:-10, 24:-24, :]
    # Get the base name of the image (e.g. "image.png")
    base_name = os.path.basename(img_path)

    # Get the name without the extension (e.g. "image")
    name_without_extension = os.path.splitext(base_name)[0]

    # Create the output path
    output_path = os.path.join(output_folder, name_without_extension + ".jpg")

    # Write the image in the output folder in JPEG format
    cv2.imwrite(output_path, img)

    if counter > 50:
        break

    counter += 1

print("Conversion from PNG to JPEG completed.")

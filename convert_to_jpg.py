import cv2
import glob
import os
from tqdm import tqdm

# Input and output folders
input_folder = "/home/anaya/Development/tfm_project/datasets/dataset_carla_004_town04_2/images"
output_folder = "/home/anaya/Development/tfm_project/datasets/dataset_carla_004_town04_2/images2"

# If output_folder doesn't exist, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all .png images in the input folder
input_images = glob.glob(os.path.join(input_folder, "*.jpg"))

# Iterate over all input images
for img_path in tqdm(input_images, desc="Converting images", unit="image"):
    # Read the image in color (default setting)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # resize to w:320, h:180
    img = cv2.resize(img, (320, 180))

    # crop the top 20 pixels to convert from 320x180 to 320x160
    img = img[20:, :, :]

    # Get the base name of the image (e.g. "image.png")
    base_name = os.path.basename(img_path)

    # Get the name without the extension (e.g. "image")
    name_without_extension = os.path.splitext(base_name)[0]

    # Create the output path
    output_path = os.path.join(output_folder, name_without_extension + ".jpg")

    # Write the image in the output folder in JPEG format
    cv2.imwrite(output_path, img)

print("Conversion from PNG to JPEG completed.")

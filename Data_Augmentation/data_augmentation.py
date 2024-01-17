import cv2
import numpy as np
import os
from mytransform import rotate, cropping, day2night, add_rain

# Paths of the original and transformed images
source_folder = './data/original_data'
target_folder = './data/transformed_data'

# Create target folder if not exists.
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Load all image files from the folder.
for file_name in os.listdir(source_folder):
    # If a different format is needed, a new function must be implemented.
    if file_name.endswith('.jpg'):
        img_path = os.path.join(source_folder, file_name)
        img = cv2.imread(img_path)

        name = file_name[:-4]
        extension = file_name[-4:]

        # Darken images by 0.3, 0.5, 0.7
        darked03 = day2night(img, 0.3)
        cv2.imwrite(os.path.join(target_folder, name + '_d03' + extension), darked03)
        print('Darken image by 0.3 is complete.')

        darked05 = day2night(img, 0.5)
        cv2.imwrite(os.path.join(target_folder, name + '_d05' + extension), darked05)
        print('Darken image by 0.5 is complete.')

        darked07 = day2night(img, 0.7)
        cv2.imwrite(os.path.join(target_folder, name + '_d07' + extension), darked07)
        print('Darken image by 0.7 is complete.')

        # Add rain to image
        rainy = add_rain(img)
        cv2.imwrite(os.path.join(target_folder, name + '_rn' + extension), rainy)
        print('Adding rain is complete.')

        # Rotate Image
        angle = np.random.randint(-15, 15)
        rotated = rotate(img, angle)
        cv2.imwrite(os.path.join(target_folder, name + '_rt' + extension), rotated)
        print('Rotation is complete.')

        # Crop Image
        height, width = img.shape[0], img.shape[1]
        start_y = int(np.random.uniform(0, 0.2) * height)
        start_x = int(np.random.uniform(0, 0.2) * width)
        end_y = int(np.random.uniform(0.8, 1) * height)
        end_x = int(np.random.uniform(0.8, 1) * width)
        cropped = cropping(img, start_x, start_y, end_x, end_y)
        cv2.imwrite(os.path.join(target_folder, name + '_cr' + extension), cropped)
        print('Cropping is complete.')

print("All Images have been transformed.")


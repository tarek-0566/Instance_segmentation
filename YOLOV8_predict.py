# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:59:53 2024

@author: islam
"""


# =============================================================================
# from ultralytics import YOLO
# import numpy as np
# import cv2
# 
# model_path = 'C:/Users/islam/Desktop\Model/1000_epochs_253_image_train_22_image_validation/best.pt'
# image_path = 'C:/Users/islam/Desktop/test_daten/P7100678.jpg'
# 
# img = cv2.imread(image_path)
# print(img.shape)
# H, W, _ = img.shape
# 
# model = YOLO(model_path)
# 
# results = model(img)  # , conf=0.25, iou=0.45)
# 
# # Create a blank image for combined masks
# combined_mask = np.zeros((H, W), dtype=np.uint8)
# 
# for i, result in enumerate(results):
#     if result.masks is not None:
#         for j, mask in enumerate(result.masks.data):
#             mask = mask.numpy() * 255
#             mask = cv2.resize(mask, (W, H))
#             combined_mask = np.maximum(combined_mask, mask)  # Combine masks
# 
# cv2.imwrite('./P7100678_pred.png', combined_mask)
# print('Saved combined mask to ./P7100678_pred.png')
# =============================================================================

import os
from ultralytics import YOLO
import numpy as np
import cv2

# Paths
model_path = 'C:/Users/islam/Desktop/Model/1000_epochs_253_image_train_22_image_validation/best.pt'
input_folder = 'C:/Users/islam/Desktop/test_date/'
output_folder = 'C:/Users/islam/Desktop/output_masks/'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load model
model = YOLO(model_path)

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
        image_path = os.path.join(input_folder, image_name)
        
        # Read image
        img = cv2.imread(image_path)
        H, W, _ = img.shape
        
        # Perform prediction
        results = model(img)  # , conf=0.25, iou=0.45)
        
        # Create a blank image for combined masks
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Process results
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    mask = mask.numpy() * 255
                    mask = cv2.resize(mask, (W, H))
                    combined_mask = np.maximum(combined_mask, mask)  # Combine masks
        
        # Save combined mask
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_pred.png")
        cv2.imwrite(output_path, combined_mask)
        print(f'Saved combined mask to {output_path}')

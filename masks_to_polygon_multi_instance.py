# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:42:46 2024

@author: islam
"""

import os
import cv2
import numpy as np

input_dir = 'C:/Users/islam/Downloads/task_a and b detection_annotations_2024_07_05_14_36_59_segmentation mask 1.1/SegmentationClass'
output_dir = 'C:/Users/islam/Downloads/task_a and b detection_annotations_2024_07_05_14_36_59_segmentation mask 1.1/SegmentationLabel'

def get_class_label(color):
    # Define color thresholds for "A" (red) and "B" (shady green)
    red_lower = np.array([0, 0, 100])
    red_upper = np.array([50, 50, 255])
    green_lower = np.array([0, 50, 0])
    green_upper = np.array([50, 255, 50])
    
    if np.all(color >= red_lower) and np.all(color <= red_upper):
        return 0  # Class "A"
    elif np.all(color >= green_lower) and np.all(color <= green_upper):
        return 1  # Class "B"
    else:
        return None  # Undefined class

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # Load the mask image
    mask = cv2.imread(image_path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape[:2]
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append((cnt, polygon))

    # Write polygons with class labels
    with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
        for cnt, polygon in polygons:
            # Get the color of the mask at the centroid of the contour
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            class_label = get_class_label(mask[cY, cX])
            
            if class_label is not None:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('{} {} '.format(class_label, p))
                    else:
                        f.write('{} '.format(p))
        f.close()

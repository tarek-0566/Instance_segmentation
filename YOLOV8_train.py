# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:07:40 2024

@author: islam
"""

DATA_DIR = 'C:/Users/islam/Desktop/Data'


import os

from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='C:/Users/islam/Downloads/config.yaml', epochs=1000, imgsz=640, patience=1010)
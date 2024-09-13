# YOLOv8 Project

This project includes scripts for training and predicting using the YOLOv8 model, as well as utilities for converting masks to polygons for both single and multi-instance segmentation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Masks to Polygon Conversion](#masks-to-polygon-conversion)
    - [Single Instance](#single-instance)
    - [Multi Instance](#multi-instance)
- [Configuration](#configuration)
- [Files Description](#files-description)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yolov8-project.git
    cd yolov8-project
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the YOLOv8 model, use the `YOLOV8_train.py` script. Ensure that your dataset and configuration are properly set up.

```sh
python YOLOV8_train.py --config config.yaml
```

### Prediction

To make predictions using a trained YOLOv8 model, use the `YOLOV8_predict.py` script.

```sh
python YOLOV8_predict.py --config config.yaml --image_path path/to/image.jpg
```

### Masks to Polygon Conversion

#### Single Instance

For converting single instance masks to polygons, use the `masks_to_polygon.py` script.

```sh
python masks_to_polygon.py --mask_path path/to/mask.png --output_path path/to/output.json
```

#### Multi Instance

For converting multi-instance masks to polygons, use the `masks_to_polygon_multi_instance.py` script.

```sh
python masks_to_polygon_multi_instance.py --mask_path path/to/mask.png --output_path path/to/output.json
```

## Configuration

The `config.yaml` file contains all the configuration parameters for training and prediction. Modify this file to suit your dataset and model requirements.

Example `config.yaml`:

```yaml
model:
  name: yolov8
  input_size: 640
  epochs: 50
  batch_size: 16
  learning_rate: 0.001

dataset:
  train: path/to/train
  val: path/to/val
  test: path/to/test

output:
  model_save_path: path/to/save/model
  prediction_save_path: path/to/save/predictions
```

## Files Description

- `YOLOV8_train.py`: Script for training the YOLOv8 model.
- `YOLOV8_predict.py`: Script for making predictions using the trained YOLOv8 model.
- `config.yaml`: Configuration file for setting parameters for training and prediction.
- `masks_to_polygon.py`: Utility script to convert single instance masks to polygons.
- `masks_to_polygon_multi_instance.py`: Utility script to convert multi-instance masks to polygons.

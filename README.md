# RoboWeedMaPS Dataset Tools

A toolkit for extracting, processing, and training on the RoboWeedMaPS (RWM) dataset created and managed by the "Computer Vision in Biosystems" research group from Aarhus University. This repository provides utilities for creating plant detection datasets from the RoboWeedMaPS database and training various computer vision models.

## Overview 

RoboWeedMaPS (RWM) is a large dataset of agricultural images with annotations for crops and weeds. This toolkit extracts training data from the RWM database and prepares it for training YOLO object detection models, with support for both YOLOv5 and YOLOv11 formats.

Key features:
- Database connection and query tools for the RWM SQL database
- Extraction of images and annotations based on training flags
- Special handling for PSEZ (Plant Stem Emergence Zone) annotations
- Support for multiple output formats (YOLOv5, YOLOv11)
- Dataset verification and quality checks
- Training scripts for YOLOv11

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rwm_dataset_tools.git
cd rwm_dataset_tools

# Install dependencies
pip install -r requirements.txt
```

## Dataset Extraction

The main functionality is to extract a YOLO-compatible dataset from the RWM database:
```bash
# Basic extraction with default settings (YOLOv11 format)
python run.py

# Extract to a specific output directory
python run.py --output-dir /path/to/output --format yolov11

# Extract with image copying instead of symlinks (useful for fast storage)
python run.py --output-base-dir /fast_data --copy-images

# Test database connection before extraction
python run.py --debug-db --dry-run
```

## Configuration

The extraction process is controlled by YAML configuration files in the `config/` directory:
- `default.yaml`: Base configuration with database settings and dataset parameters
- `models/yolov5.yaml`: YOLOv5-specific settings
- `models/yolov11.yaml`: YOLOv11-specific settings

These files can customized to control:
- Database connection parameters
- Training/validation/test split ratios
- Special handling for fixed image sets
- EPPO codes for class labels
- Output directory structure

## Training

After extracting the dataset, a YOLO model can trained using the scripts in the `training/` directory:

```bash
# Test that the dataset can be loaded correctly
cd training/scripts
python test_extraction.py --dataset /path/to/dataset.yaml

# Train a YOLOv11 model (quick test)
python train_yolov11.py --epochs 1 --img-size 640 --batch-size 12 --name test_run

# Full training
python train_yolov11.py --epochs 600 --img-size 2048 --batch-size 1 --device 0 --name rwm_yolov11
```

## Dataset Structure

The extracted dataset follows the standard YOLO structure:

```bash
/path/to/output/
├── dataset.yaml        # Dataset configuration for YOLO
├── images/
│   ├── train/         # Training images
│   ├── val/           # Validation images
│   └── test/          # Test images
└── labels/
    ├── train/         # Training annotations
    ├── val/           # Validation annotations
    └── test/          # Test annotations
```

Each label file contains annotations in YOLO format:

```bash
<class_id> <x_center> <y_center> <width> <height>
```

## EPPO Codes and Classes

The dataset contains the follow plant species as EPPO codes, which can be used as classes for training:
- `PPPMM`: Monocot weed
- `PPPDD`: Dicot weed
- `VICFX`: Faba bean
- `PIBSA`: Field Pea
- `ZEAMX`: Maize
- `SOLTU`: Potato
- `SPQOL`: Spinach
- `BEAVA`: Sugar beet
- `CIRAR`: Creeping Thistle
- `BRSOL`: White cabbage
- `FAGES`: Buckwheat
- `1LUPG`: Lupinus
- `CHEAL`: Fat-hen
- `FUMOF`: Common fumitory
- `1MATG`: Chamomile
- `GERMO`: Dovesfoot cranesbill
- `EPHHE`: Sun spurge
- `EQUAR`: Field horsetail
- `GALAP`: Cleavers
- `1CRUF`: Crucifer
- `SINAR`: Charlock
- `POLAV`: Pale persicaria
- `VERPE`: Common speedwell
- `VIOAR`: Field pansy
- `POLCO`: Wild buckwheat
- `TAROF`: Dandelion
- `POLLA`: Black bindweed
- `ATXPA`: Common orache
- `LAMPU`: Red dead-nettle
- `SENVU`: Common groundsel
- `PSEZ`: Plant Stem Emergence Zone
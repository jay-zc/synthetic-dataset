# synthetic-dataset

A synthetic dataset generator for salmon lice detection using computer vision techniques.

## Overview

Two main components:
1. **Instance Extraction** (`crop_instances.py`): Extracts salmon lice instances using SAM
2. **Synthetic Dataset Generation** (`generate.py`): Creates synthetic images by combining extracted instances with backgrounds

## Usage

### Extract Instances
```bash
python crop_instances.py
```

### Generate Synthetic Dataset
```bash
python generate.py
```

## File Structure

```
synthetic-dataset/
├── crop_instances.py         # Instance extraction script
├── generate.py               # Synthetic dataset generation script
├── genertae_single_image.py  # Single image generation utility
├── optimal_exp.csv           # Experiment parameters
├── data/
│   ├── images/               # Original annotated images
│   ├── labels/               # YOLO format annotations
│   ├── transparent_checked/  # Salmon lice instances (class 0,1)
│   ├── impurities/           # Impurity instances (class 3)
│   └── pure_background_v1/   # Clean background images
```

## Naming Convention

- **Class 0**: Nauplius stage salmon lice
- **Class 1**: Copepodite stage salmon lice  
- **Class 3**: Impurities/debris

Examples: `10_0_1.png`: Instance 0 from image 10, class 1 (Copepodite)

## Requirements

```bash
pip install opencv-python numpy pillow pyyaml torch torchvision segment-anything
```

## SAM pretrained checkpoint

* `Pretrained Checkpoint or Demo Files` : 

    * `sam_vit_b_01ec64.pth`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/1oqxAzp_7qOJb6IZ4GffWdA). |  
    * `sam_vit_h_4b8939.pth`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/13zZWESnZsB2c9QdzEe15cQ). |  
    * `sam_vit_l_0b3195.pth`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/18kpjceAx5TEpPerKnSOW0A). | 

## Features

- Multi-threaded parallel generation
- Flexible augmentation (scaling, rotation, blur, lighting)
- YOLO compatible annotations

## Troubleshooting

1. **CUDA Memory Issues**: Reduce batch size or use smaller SAM model
2. **File Path Errors**: Ensure all directories exist
3. **Empty Annotations**: Verify YOLO format annotations
4. **Transparent Images**: Check PNG alpha channel

## Acknowledgments

This project uses [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for instance segmentation. We thank the Facebook Research team for making this powerful tool available to the community.

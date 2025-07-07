# synthetic_dataset_generator.py

import os
import sys
import csv
import yaml
import random
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import cv2
import numpy as np

# Define the postfix for image files
POSTFIX = '.jpg'

def Paste_mask(
    image_path=None, 
    anno_path=None, 
    transparent_image_path=None, 
    impurities_path=None,
    pure_background_path=None, 
    AUG_DST_DIR=None, 
    Ratio_of_pure_background=1.0,
    REPEATE=1, 
    Num_of_pasted_objects=0,
    Scale_of_pobjects=(1.0, 1.0),
    Random_orientation=True,
    Num_of_impurities=100,
    Contrast=1,
    Brightness=0,
    Motion_blur=1,
    Gaussian_blur=1,
    postfix='.jpg',
    exp_id=0,
    use_raw_image_during_augmentation=False
):
    """
    Generates synthetic images by pasting objects and impurities onto backgrounds.

    Parameters:
    - image_path: Path to original images.
    - anno_path: Path to original annotations.
    - transparent_image_path: Path to transparent object images (.png).
    - impurities_path: Path to impurity images (.png).
    - pure_background_path: Path to pure background images (.jpg).
    - AUG_DST_DIR: Destination directory for synthetic dataset.
    - Ratio_of_pure_background: Ratio of pure background images to use.
    - REPEATE: Number of repetitions per image.
    - Num_of_pasted_objects: Maximum number of objects to paste per image.
    - Scale_of_pobjects: Tuple indicating the scale range for objects.
    - Random_orientation: Whether to apply random orientation to objects.
    - Num_of_impurities: Maximum number of impurities to add.
    - Contrast: Contrast adjustment factor.
    - Brightness: Brightness adjustment factor.
    - Motion_blur: Maximum kernel size for motion blur.
    - Gaussian_blur: Maximum kernel size for Gaussian blur.
    - postfix: File extension for output images.
    - exp_id: Experiment identifier.
    - use_raw_image_during_augmentation: Whether to include original images during augmentation.
    """
    # Create necessary directories
    os.makedirs(AUG_DST_DIR, exist_ok=True)
    os.makedirs(os.path.join(AUG_DST_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(AUG_DST_DIR, 'labels'), exist_ok=True)

    # Write experiment info to info.txt
    info_path = os.path.join(AUG_DST_DIR, "info.txt")
    with open(info_path, 'w') as out_file:
        out_file.write(
            f'exp_id = {exp_id}\n' +
            f'image_path = {image_path}\n' +
            f'anno_path = {anno_path}\n' + 
            f'transparent_image_path = {transparent_image_path}\n' +
            f'impurities_path = {impurities_path}\n' +
            f'pure_background_path = {pure_background_path}\n' +
            f'AUG_DST_DIR = {AUG_DST_DIR}\n' + 
            f'Ratio_of_pure_background = {Ratio_of_pure_background}\n' +
            f'REPEATE = {REPEATE}\n' +
            f'Num_of_pasted_objects = {Num_of_pasted_objects}\n' +
            f'Scale_of_pobjects = {Scale_of_pobjects}\n' +
            f'Random_orientation = {Random_orientation}\n' +
            f'Num_of_impurities = {Num_of_impurities}\n' +
            f'Contrast = {Contrast}\n' +
            f'Brightness = {Brightness}\n' +
            f'Motion_blur = {Motion_blur}\n' +
            f'Gaussian_blur = {Gaussian_blur}\n'
        )

    # Load transparent object and impurity images
    transparent_paths = get_image_paths(transparent_image_path, postfix='.png')  
    impurities_paths = get_image_paths(impurities_path, postfix='.png')  

    if image_path or pure_background_path:
        img_paths, annos = get_dataset(anno_path, image_path)
        dataset_size = len(img_paths)

        back_image = get_image_paths(pure_background_path, postfix='.jpg')
        num_pure_back = int(dataset_size * Ratio_of_pure_background)
        random_selection = random.sample(back_image, min(len(back_image), num_pure_back))      

        if use_raw_image_during_augmentation:
            img_paths_extended = img_paths + random_selection
        else:
            img_paths_extended = random_selection

        for i in range(len(img_paths_extended)):
            if i < len(annos) and use_raw_image_during_augmentation:
                raw_bbox = Get_bbox(annos[i])
            else:
                raw_bbox = []

            for j in range(REPEATE):
                output_img = cv2.imread(img_paths_extended[i])
                new_annos = raw_bbox.copy()

                num_objects = random.randint(1, Num_of_pasted_objects)
                num_impurities = random.randint(0, Num_of_impurities)

                # Paste objects
                for _ in range(num_objects):
                    idx = random.randint(0, len(transparent_paths)-1)
                    transparent = transparent_paths[idx]
                    transparent_image = cv2.imread(transparent, cv2.IMREAD_UNCHANGED)
                    parts = os.path.basename(transparent).split('_')[-1]
                    class_id = int(os.path.splitext(parts)[0])
                    output_img, obj = paste_image_with_transform(
                        output_img,
                        transparent_image,
                        class_id,
                        Scale_of_pobjects,
                        Random_orientation
                    )
                    new_annos.append(obj)

                # Add impurities
                for _ in range(num_impurities):
                    idx = random.randint(0, len(impurities_paths)-1)
                    impurity = impurities_paths[idx]
                    impurity_img = cv2.imread(impurity, cv2.IMREAD_UNCHANGED)
                    output_img, _ = paste_image_with_transform(output_img, impurity_img)

                # Adjust illumination
                output_img = adjust_illumination(
                    output_img, 
                    alpha=random.randint(1, Contrast), 
                    beta=random.randint(0, Brightness)
                )

                # Apply Gaussian blur
                gaussian_k = random.randint(1, Gaussian_blur)
                if gaussian_k % 2 == 0:
                    gaussian_k -= 1
                if gaussian_k > 1:
                    output_img = cv2.GaussianBlur(output_img, ksize=(gaussian_k, gaussian_k), sigmaX=0, sigmaY=0)

                # Apply motion blur
                motion_k = random.randint(1, Motion_blur)
                if motion_k > 1:
                    output_img = motion_blur(output_img, kernel_size=motion_k)

                # Save the synthetic image
                new_img_name = os.path.join(AUG_DST_DIR, 'images', f"{i}_{j}{postfix}")
                cv2.imwrite(new_img_name, output_img)
                print(f"Thread {exp_id}: Writing {new_img_name}")

                # Save the annotation
                new_anno_name = os.path.join(AUG_DST_DIR, 'labels', f"{i}_{j}.txt")
                with open(new_anno_name, 'w') as anno_file:
                    for bbox in new_annos:
                        anno_file.write(" ".join(map(str, bbox)) + '\n')

def paste_image_with_transform(background, 
                               transparent_image, 
                               class_id =0,
                               Scale_of_pobjects=(0.8,1.2),
                               Random_orientation=True):
    #start = time.time() 
    back_height, back_width = background.shape[:2]

    #scale = random.uniform(0.8, 1.5)
    scale = random.uniform(Scale_of_pobjects[0],Scale_of_pobjects[1])
    angle = random.uniform(0, 1) *365*Random_orientation
    #scale=1
    #angle=0


    # Load the background image and the transparent image
    old_height, old_width = transparent_image.shape[:2]
    pad_x = int(0.2*old_width)
    pad_y = int(0.2*old_height)
    transparent_image = cv2.copyMakeBorder(transparent_image, pad_x, pad_y, pad_x, pad_y, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    
    # Extract the alpha channel from the transparent image


    # Get the height and width of the transparent image
    height, width = transparent_image.shape[:2]


    # Calculate the new dimensions after scaling
    new_width, new_height = int(width * scale), int(height * scale)

    # Define the position to paste the transparent image on the background
    x = int((back_width - new_width)*random.uniform(0, 0.95))
    y = int((back_height- new_height)*random.uniform(0, 0.95))


    # Resize the transparent image and alpha channel to the new dimensions
    transparent_image = cv2.resize(transparent_image, (new_width, new_height))


    # Create a copy of the background to paste onto
   
    #result = background.copy()
    result = background
 
    # Rotate the transparent image and the mask
    M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1.0)
    transparent_image = cv2.warpAffine(transparent_image, M, (new_width, new_height))
    alpha_channel = transparent_image[:, :, 3]
    alpha_channel = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
    # alpha_channel = cv2.warpAffine(alpha_channel, M, (new_width, new_height))


    non_zero_indices = np.nonzero(alpha_channel)     #  time consuming


    # Calculate the bounds
    left = np.min(non_zero_indices[1])

    right = np.max(non_zero_indices[1])
    top = np.min(non_zero_indices[0])
    bottom = np.max(non_zero_indices[0])
    alpha_channel = alpha_channel / 255.0


      
    # Paste the transparent image onto the background at the specified position
    result[y:y+new_height, x:x+new_width] = result[y:y+new_height, x:x+new_width] * (1 - alpha_channel) + \
        transparent_image[:, :, :3] * (alpha_channel)


    xc = x+0.5*(left + right)
    yc = y + 0.5*(top + bottom)
    w = right-left
    h = bottom - top
    h_b, w_b = result.shape[:2]
    object = [class_id, xc/w_b, yc/h_b, w/w_b, h/h_b]

    return result, object

def adjust_illumination(image, alpha=1.0, beta=0):
    """
    Adjusts the contrast and brightness of an image.

    Parameters:
    - image: Input image.
    - alpha: Contrast factor.
    - beta: Brightness factor.

    Returns:
    - Adjusted image.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def motion_blur(image, kernel_size=25):
    """
    Applies motion blur to an image.

    Parameters:
    - image: Input image.
    - kernel_size: Size of the motion blur kernel.

    Returns:
    - Blurred image.
    """
    # Generate a horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size

    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def get_dataset(annotation_folder, image_folder):
    """
    Retrieves image and annotation file paths.

    Returns:
    - List of image paths.
    - List of corresponding annotation paths.
    """
    image_paths = []
    annotation_paths = []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(POSTFIX)]
    annotation_files = [f for f in os.listdir(annotation_folder) if f.lower().endswith('.txt')]

    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        annotation_file = f"{image_id}.txt"
        if annotation_file in annotation_files:
            image_paths.append(os.path.join(image_folder, image_file))
            annotation_paths.append(os.path.join(annotation_folder, annotation_file))

    return image_paths, annotation_paths

def get_image_paths(image_folder, postfix='.png'):
    """
    Retrieves all image file paths with the specified postfix.

    Returns:
    - List of image paths.
    """
    image_paths = []
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(postfix):
            image_paths.append(os.path.join(image_folder, file_name))
    return image_paths

def Get_bbox(annotation_file, w=0, h=0, xyxy=False):
    """
    Extracts bounding boxes from a YOLO-format annotation file.

    Parameters:
    - annotation_file: Path to the annotation file.
    - w: Width of the image (optional for xyxy format).
    - h: Height of the image (optional for xyxy format).
    - xyxy: If True, converts from YOLO to xyxy format.

    Returns:
    - List of bounding boxes.
    """
    objects = []
    with open(annotation_file, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.strip().split(' ')
            if xyxy:
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                wc = float(parts[3]) * w
                hc = float(parts[4]) * h

                x1 = xc - 0.5 * wc
                x2 = xc + 0.5 * wc
                y1 = yc - 0.5 * hc
                y2 = yc + 0.5 * hc

                objects.append([int(parts[0]), x1, y1, x2, y2])
            else:
                objects.append([
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4])
                ])
    return objects

def process_csv_row(row, AUG_DST_DIR, dict_params, paths):
    """
    Processes a single CSV row to generate synthetic dataset.

    Parameters:
    - row: List of CSV row values.
    - AUG_DST_DIR: Base destination directory.
    - dict_params: Dictionary for scaling parameters.
    - paths: Tuple containing necessary paths.

    Returns:
    - None
    """
    try:
        # Parse parameters from CSV row
        Ratio_of_pure_background = float(row[0])
        Dataset_magnification = int(row[1])
        Num_of_pasted_objects = int(row[2])
        Scale_of_pobjects = dict_params.get(row[3], (1.0, 1.0))
        Random_orientation = row[4].strip().lower() in ['true', '1', 'yes']
        Num_of_impurities = int(row[5])
        Contrast = int(row[6])
        Brightness = int(row[7])
        Motion_blur = int(row[8])
        Gaussian_blur = int(row[9])
        exp_id = row[11]
        DST_folder = os.path.join(AUG_DST_DIR, exp_id)

        print(f"Thread {exp_id}: Starting dataset generation.")

        # Create YAML file for the experiment
        yaml_data = {
            'train': os.path.join(DST_folder, 'images'),
            'val': 'data/images/val',
            'test': 'data/images/test2',
            'nc': 2,
            'names': ['Nauplius', 'Copepodite']
        }
        yaml_dir = os.path.join(AUG_DST_DIR, 'yaml')
        os.makedirs(yaml_dir, exist_ok=True)
        yaml_file_path = os.path.join(yaml_dir, f"{exp_id}.yaml")
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        # Generate synthetic dataset
        Paste_mask(
            image_path=paths['image_path'], 
            anno_path=paths['anno_path'], 
            transparent_image_path=paths['transparent_image_path'], 
            impurities_path=paths['impurities_path'],
            pure_background_path=paths['pure_background_path'], 
            AUG_DST_DIR=DST_folder,
            Ratio_of_pure_background=Ratio_of_pure_background, 
            REPEATE=Dataset_magnification, 
            Num_of_pasted_objects=Num_of_pasted_objects,
            Scale_of_pobjects=Scale_of_pobjects,
            Random_orientation=Random_orientation,
            Num_of_impurities=Num_of_impurities,
            Contrast=Contrast,
            Brightness=Brightness,
            Motion_blur=Motion_blur,
            Gaussian_blur=Gaussian_blur,
            postfix=POSTFIX,
            exp_id=exp_id,
            use_raw_image_during_augmentation=False  # Set as needed
        )

        print(f"Thread {exp_id}: Completed dataset generation.")

    except Exception as e:
        print(f"Thread {exp_id}: Error occurred - {e}")

def main():
    # Define paths (Update these paths as per your directory structure)
    image_path = 'data/images/train'        
    anno_path = 'data/labels/train'
    transparent_image_path = 'data/transparent_checked'
    impurities_path = 'data/impurities'
    pure_background_path = 'data/pure_background_v1'
    csv_file_path = "data/exp_Dec17.csv"
    AUG_DST_DIR = 'data/synthetic_dataset'    

    # Parameter dictionary for scaling
    dict_params = {
        '1': (1, 1),
        '0.9~1.1': (0.9, 1.1),
        '0.8~1.2': (0.8, 1.2)
    }

    # Load all necessary paths once
    paths = {
        'image_path': image_path,
        'anno_path': anno_path,
        'transparent_image_path': transparent_image_path,
        'impurities_path': impurities_path,
        'pure_background_path': pure_background_path
    }

    # Read CSV and collect all rows
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip header
        rows = [row for row in reader if row]  # Ensure no empty rows

    print(f"Total experiments to run: {len(rows)}")

    # Define the number of threads (adjust as needed)
    max_workers = min(18, os.cpu_count() or 1)  # For example, use up to 8 threads
    max_workers = os.cpu_count() or 1
    

    # Use ThreadPoolExecutor to process CSV rows in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_csv_row, row, AUG_DST_DIR, dict_params, paths)
            for row in rows
        ]

        # Optionally, track progress
        for future in as_completed(futures):
            pass  # You can implement logging or progress tracking here

    print("All synthetic dataset generations completed.")

if __name__ == "__main__":
    main()

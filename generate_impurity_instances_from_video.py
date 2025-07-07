from segment_anything import SamPredictor
from segment_anything import sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from PIL import Image
import os
import random
import math
import time
import glob

DEVICE = "cuda"
CHECKPOINT = ["/home/jayc/jayc/synthetic/synthetic_main/Data_synthesis_tool/Jayc_SAM_Checkpoints/sam_vit_b_01ec64.pth","/home/jayc/jayc/synthetic/synthetic_main/Data_synthesis_tool/Jayc_SAM_Checkpoints/sam_vit_l_0b3195.pth", "/home/jayc/jayc/synthetic/synthetic_main/Data_synthesis_tool/Jayc_SAM_Checkpoints/sam_vit_h_4b8939.pth"]
MODEL_TYPE = ["vit_b", "vit_l", "vit_h"]
postfix = '.jpg' 

def generate_random_boxes(image_width, image_height, num_boxes=3, min_size=50, max_size_ratio=0.3):
    """
    Generate random bounding boxes
    
    Args:
    - image_width: Width of the image
    - image_height: Height of the image  
    - num_boxes: Number of bounding boxes to generate
    - min_size: Minimum size of bounding boxes
    - max_size_ratio: Maximum size ratio of bounding boxes relative to image
    
    Returns:
    - List of bounding boxes in format [[class_id, x1, y1, x2, y2], ...]
    """
    box_list = []
    max_width = int(image_width * max_size_ratio)
    max_height = int(image_height * max_size_ratio)
    
    for i in range(num_boxes):
        # Randomly generate bounding box width and height
        box_width = random.randint(min_size, min(max_width, image_width - min_size))
        box_height = random.randint(min_size, min(max_height, image_height - min_size))
        
        # Randomly generate top-left coordinates
        x1 = random.randint(0, image_width - box_width)
        y1 = random.randint(0, image_height - box_height)
        
        # Calculate bottom-right coordinates
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        # Set class ID to 0 (since we don't care about specific classes)
        box_list.append([0, x1, y1, x2, y2])
    
    return box_list

def save_masked_region(image, mask, output_path=None, min_brightness_threshold=30, min_non_black_ratio=0.1):
    """
    Save masked region as transparent image given an image and its mask
    
    Args:
    - image: Input image
    - mask: Mask array
    - output_path: Save path (optional)
    - min_brightness_threshold: Minimum average brightness threshold (0-255)
    - min_non_black_ratio: Minimum ratio of non-black pixels (0-1)
    
    Returns:
    - Transparent image array or None (if filtered out)
    """
    # Convert tensor to numpy array
    mask_array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    mask_array = mask_array.astype(np.uint8) * 255

    # Create mask for region of interest
    masked_image = cv2.bitwise_and(image, image, mask=mask_array)

    # Find contours and get bounding rectangle
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract the masked region
    cropped_region = image[y:y+h, x:x+w]
    cropped_mask = mask_array[y:y+h, x:x+w]
    
    # Only analyze pixels inside the mask
    masked_pixels = cropped_region[cropped_mask > 0]
    
    if len(masked_pixels) == 0:
        return None
    
    # Calculate average brightness of the masked region
    avg_brightness = np.mean(cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY))
    
    # Calculate the ratio of non-black pixels
    # Define black pixels as those with all channel values less than 20
    non_black_pixels = np.sum(np.all(masked_pixels > 20, axis=1))
    non_black_ratio = non_black_pixels / len(masked_pixels)
    
    # Filter condition: discard if brightness is too low or non-black pixel ratio is too low
    if avg_brightness < min_brightness_threshold or non_black_ratio < min_non_black_ratio:
        return None
    
    # Copy masked region to transparent background
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
    transparent_image[..., :3] = cropped_region
    transparent_image[..., 3] = cropped_mask

    return transparent_image

def save_masks(image_dir, output_dir, num_boxes_per_image=3, min_brightness_threshold=30, min_non_black_ratio=0.1):
    """
    Use random bounding boxes as prompts to generate and save masked regions
    
    Args:
    - image_dir: Path to input image folder
    - output_dir: Path to output folder for saving results
    - num_boxes_per_image: Number of random bounding boxes generated per image
    - min_brightness_threshold: Minimum average brightness threshold, images below this value will be filtered out
    - min_non_black_ratio: Minimum ratio of non-black pixels, images below this value will be filtered out
    """
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # Load model
    sam = sam_model_registry[MODEL_TYPE[0]](checkpoint=CHECKPOINT[0])

    # Move model to GPU
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # Get image path list
    image_path_list = get_image_paths(image_dir, postfix)

    # Statistics
    total_generated = 0
    total_saved = 0
    total_filtered = 0

    # Process each image in the dataset
    for image_idx, image_path in enumerate(image_path_list):
        image = cv2.imread(image_path)
        print(f"Processing: {image_path}")
        
        # Convert color format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = rgb_image.shape[:2]   
        
        # Generate random bounding boxes
        random_box_list = generate_random_boxes(image_width, image_height, num_boxes_per_image)
        
        # Convert to tensor format
        input_boxes = torch.tensor(random_box_list, device=predictor.device)

        # Apply transformation to bounding boxes
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes[:,1:], rgb_image.shape[:2])
        predictor.set_image(rgb_image)
        
        # Predict masks
        mask_list, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            )
            
        # Save each masked region (with filtering)
        for mask_idx, mask in enumerate(mask_list):
            total_generated += 1
            class_id = random_box_list[mask_idx][0]
            
            # Generate transparent image with brightness filtering
            transparent_image = save_masked_region(
                image, 
                mask[0], 
                min_brightness_threshold=min_brightness_threshold,
                min_non_black_ratio=min_non_black_ratio
            )
            
            if transparent_image is not None:
                # Only save images that pass the filter
                output_path = os.path.join(output_dir, f'{image_idx}_{mask_idx}_{class_id}.png')
                cv2.imwrite(output_path, transparent_image)
                total_saved += 1
            else:
                total_filtered += 1

    # Output statistics
    print(f"\n=== Extraction Results Statistics ===")
    print(f"Total generated: {total_generated} regions")
    print(f"Successfully saved: {total_saved} regions")
    print(f"Filtered out: {total_filtered} regions (Percentage: {total_filtered/total_generated*100:.1f}%)")
    print(f"Save rate: {total_saved/total_generated*100:.1f}%")

def get_image_paths(image_folder, suffix):
    """
    Get all image paths with specified suffix in folder
    
    Args:
    - image_folder: Image folder path
    - suffix: Image file suffix
    
    Returns:
    - List of image paths
    """
    image_path_list = []
    image_file_list = [f for f in os.listdir(image_folder) if f.lower().endswith(suffix)]
    for image_file in image_file_list:
        image_path = os.path.join(image_folder, image_file)
        image_path_list.append(image_path)
    return image_path_list

# Keep original functions as backup (commented out)
"""
def get_dataset(annotation_folder, image_folder):
    # This function is no longer needed since we don't use annotations
    pass

def Get_bbox(filename,w=0,h=0, xyxy=False):
    # This function is no longer needed since we use random bounding boxes
    pass
"""

def extract_and_process_video(video_path, output_dir, frame_interval=30, num_boxes_per_frame=5, min_brightness_threshold=30, min_non_black_ratio=0.1):
    """
    Extract frames from video at fixed intervals and process them
    
    Args:
    - video_path: Path to the video file
    - output_dir: Output directory
    - frame_interval: Interval between extracted frames
    - num_boxes_per_frame: Number of random regions generated per frame
    - min_brightness_threshold: Minimum average brightness threshold
    - min_non_black_ratio: Minimum ratio of non-black pixels
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load SAM model
    sam = sam_model_registry[MODEL_TYPE[0]](checkpoint=CHECKPOINT[0])
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n=== Video Information ===")
    print(f"Total frames: {total_frames}")
    print(f"Frame rate: {fps}")
    print(f"Estimated frames to process: {total_frames // frame_interval}")

    # Statistics
    total_generated = 0
    total_saved = 0
    total_filtered = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame_interval frames
        if frame_count % frame_interval == 0:
            print(f"\nProcessing frame {frame_count}")
            
            # Convert color format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_height, image_width = rgb_frame.shape[:2]

            # Generate random bounding boxes
            random_box_list = generate_random_boxes(image_width, image_height, num_boxes_per_frame)
            
            # Convert to tensor format
            input_boxes = torch.tensor(random_box_list, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes[:,1:], rgb_frame.shape[:2])
            predictor.set_image(rgb_frame)

            # Predict masks
            mask_list, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Save each masked region
            for mask_idx, mask in enumerate(mask_list):
                total_generated += 1
                class_id = random_box_list[mask_idx][0]

                # Generate transparent image with brightness filtering
                transparent_image = save_masked_region(
                    frame,
                    mask[0],
                    min_brightness_threshold=min_brightness_threshold,
                    min_non_black_ratio=min_non_black_ratio
                )

                if transparent_image is not None:
                    # Only save images that pass the filter
                    output_path = os.path.join(output_dir, f'frame_{frame_count:06d}_region_{mask_idx}_{class_id}.png')
                    cv2.imwrite(output_path, transparent_image)
                    total_saved += 1
                else:
                    total_filtered += 1

        frame_count += 1

    # Release resources
    cap.release()

    # Output statistics
    print(f"\n=== Extraction Results Statistics ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total regions generated: {total_generated}")
    print(f"Regions successfully saved: {total_saved}")
    print(f"Regions filtered out: {total_filtered}")
    print(f"Save rate: {total_saved/total_generated*100:.1f}%")

def process_video_folder(video_folder, output_base_dir, frame_interval=30, num_boxes_per_frame=5, min_brightness_threshold=30, min_non_black_ratio=0.1):
    """
    Process all video files in a folder
    
    Args:
    - video_folder: Path to the folder containing videos
    - output_base_dir: Base output directory
    - frame_interval: Interval between extracted frames
    - num_boxes_per_frame: Number of random regions generated per frame
    - min_brightness_threshold: Minimum average brightness threshold
    - min_non_black_ratio: Minimum ratio of non-black pixels
    """
    # Supported video formats
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    print(f"\n=== Found {len(video_files)} video files ===")
    
    # Create separate output directory for each video
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, video_name)
        
        print(f"\nProcessing video: {video_name}")
        print(f"Output directory: {output_dir}")
        
        try:
            extract_and_process_video(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=frame_interval,
                num_boxes_per_frame=num_boxes_per_frame,
                min_brightness_threshold=min_brightness_threshold,
                min_non_black_ratio=min_non_black_ratio
            )
        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
            continue

if __name__ == "__main__":
    # Example usage
    video_folder = 'video_path'  # Input video folder path
    output_base_dir = 'impurities'  # Base output directory
    frame_interval = 30  # Process every 30 frames
    num_boxes_per_frame = 5  # Generate 5 random regions per frame
    
    process_video_folder(
        video_folder=video_folder,
        output_base_dir=output_base_dir,
        frame_interval=frame_interval,
        num_boxes_per_frame=num_boxes_per_frame
    )

  

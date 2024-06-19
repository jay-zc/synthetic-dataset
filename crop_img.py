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

DEVICE = "cuda"
CHECKPOINT = ["Jayc_SAM_Checkpoints/sam_vit_b_01ec64.pth","Jayc_SAM_Checkpoints/sam_vit_l_0b3195.pth", "Jayc_SAM_Checkpoints/sam_vit_h_4b8939.pth"]
MODEL_TYPE = ["vit_b", "vit_l", "vit_h"]
postfix = '.jpg' 

def Save_mask(ANNO_DIR, IMG_DIR, DST_DIR):
    if(not os.path.exists(DST_DIR)):
        os.mkdir(DST_DIR)

    # 加载模型
    sam = sam_model_registry[MODEL_TYPE[0]](checkpoint=CHECKPOINT[0])

    # 将模型移动到gpu
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # 读取有标签的数据集
    img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)

    # 遍历数据集中的图像，每一个图像都进行增强
    for i in range(len(img_paths)):
        son_img = cv2.imread(img_paths[i])
        print(img_paths[i])
        son_image = cv2.cvtColor(son_img, cv2.COLOR_BGR2RGB)
        son_height, son_width = son_image.shape[:2]   
        son_labels = Get_bbox(annos[i], son_width, son_height, xyxy=True)
        # 转换为tensor格式
        input_son_boxes = torch.tensor(son_labels, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_son_boxes[:,1:], son_image.shape[:2])
        predictor.set_image(son_image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            )
        for i_m, mask in enumerate(masks):
            # crop area from image_one and paste into image_two
            class_id=son_labels[i_m][0]
            ds = DST_DIR + '/' + str(i) + '_'+ str(i_m) + '_'+ str(class_id) +'.png'
            transparent_image = save_masked_region(son_img, mask[0])
            cv2.imwrite(ds, transparent_image)


def save_masked_region(image, mask,ds=None):

    # given a image with its mask, save the area with mask a transparent image


    #mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    mask = mask.astype(np.uint8) * 255


    # Create a mask for the region of interest
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Copy the masked region onto the transparent background
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
    transparent_image[..., :3] = image[y:y+h, x:x+w]
    transparent_image[..., 3] = mask[y:y+h, x:x+w]

    # Save the cropped region with a transparent background as PNG
    #cv2.imwrite(ds, transparent_image)
    return transparent_image


def get_dataset(annotation_folder, image_folder):

    image_paths = []
    annotation_paths = []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(postfix)]
    annotation_files = [f for f in os.listdir(annotation_folder) if f.lower().endswith('.txt')]

    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        annotation_file = image_id + ".txt"
        if annotation_file in annotation_files:
            image_path = os.path.join(image_folder, image_file)
            annotation_path = os.path.join(annotation_folder, annotation_file)
            image_paths.append(image_path)
            annotation_paths.append(annotation_path)

    return image_paths, annotation_paths


# 获取文件夹下的图像
def get_image(image_folder, postfix):
    image_paths = []
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(postfix)]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image_paths.append(image_path)
    return image_paths



def Get_bbox(filename,w=0,h=0, xyxy=False):  # filename=??.txt

    # 用于提取yolo格式的标签
    #xyxy为true时， 返回值将yolo格式转换为xyxy 格式， 为否则返回原始标签

    objects = []
    f = []
    img_w,img_h = w,h
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        if line and line!='\n':
            splitlines = line.strip().split(' ')
            object_struct = {}
            if xyxy:
                xc = float(splitlines[1])*img_w
                yc = float(splitlines[2])*img_h
                wc = float(splitlines[3])*img_w
                hc = float(splitlines[4])*img_h

                x1 = xc-0.5*wc
                x2 = xc+0.5*wc
                y1 = yc-0.5*hc  
                y2 = yc+0.5*hc

                object_struct = [int(splitlines[0]), x1, y1, x2, y2]

            else:
                object_struct = [int(splitlines[0]), float(splitlines[1]),
                                float(splitlines[2]), float(splitlines[3]),
                                float(splitlines[4])
                                ]
            objects.append(object_struct)
        else:
            break
    return objects   
    

if __name__ == "__main__":
    
    # IMG_DIR = '/home/jayc/jayc/CV/Dataset/experiment/100/train/images'    # 图像文件夹
    # ANNO_DIR = '/home/jayc/jayc/CV/Dataset/experiment/100/train/labels'  # 图像标注的文件夹
    # DST_DIR = '/home/jayc/jayc/CV/Dataset/experiment/100/transparent_img'

    IMG_DIR = '/mnt2/jayc_2T/Ha/Jayc/final/img'
    ANNO_DIR = '/mnt2/jayc_2T/Ha/Jayc/final/txt'
    DST_DIR='/mnt2/jayc_2T/Ha/Jayc/final/transparent'
    Save_mask(ANNO_DIR, IMG_DIR, DST_DIR)

  

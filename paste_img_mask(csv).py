    # 本文件根据 正交试验生成的CSV 文件进行参数设置生成合成数据集
    # image_path ： 原始图像文件夹
    # anno_path: 原始标注文件夹
    # transparent_image_path: 由crop.py 扣出的各类目标的图像数据集，。png格式
    # impurities_path: 准备粘贴到图像的杂质图像，作为噪音或者模拟不同环境
    # AUG_DST_DIR： 合成数据集保存路径
    # pure_background_path： 空白背景图路径，通过粘贴目标或杂质进行数据合成
    # csv_file_path： 正交实验生成的CSV 文件路径


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
import csv
import yaml

postfix = '.jpg' 


def Paste_mask(image_path = None, 
                anno_path = None, 
                transparent_image_path = None, 
                impurities_path=None,
                pure_background_path = None, 
                AUG_DST_DIR = None, 
                Ratio_of_pure_background = None,
                REPEATE=1, 
                Num_of_pasted_objects=0,
                Scale_of_pobjects=1,
                Random_orientation = True,
                Num_of_impurities=100,
                Contrast = 1,
                Brightness = 0,
                Motion_blur=1,
                Gaussian_blur=1,
                postfix = '.jpg',
                exp_id=0):
    # image_path: images with annotations in dataset
    # anno_path: annotation of the images
    # transparent_image_path: path of transparent images
    # pure_background_path: background images with no existed labels
    if(not os.path.exists(AUG_DST_DIR)):
        os.mkdir(AUG_DST_DIR)   
    if(not os.path.exists(AUG_DST_DIR+'/images')):
        os.mkdir(AUG_DST_DIR + '/images')
    if(not os.path.exists(AUG_DST_DIR+'/labels')):
        os.mkdir(AUG_DST_DIR + '/labels')

    with open(os.path.join(AUG_DST_DIR, "info.txt"), 'w') as out_file:
        out_file.write('exp_id = ' + exp_id + '\n' +
                        'image_path = ' + image_path + '\n' +
                        'anno_path = ' + anno_path + '\n' + 
                        'transparent_image_path = ' + transparent_image_path + '\n' +
                        'impurities_path = ' + impurities_path + '\n' +
                        'pure_background_path = ' + pure_background_path + '\n' +
                        'AUG_DST_DIR = ' + AUG_DST_DIR + '\n' + 
                        'Ratio_of_pure_background = ' + str(Ratio_of_pure_background) + '\n' +
                        'REPEATE = ' + str(REPEATE) + '\n' +
                        'Num_of_pasted_objects = ' + str(Num_of_pasted_objects) + '\n' +
                        'Scale_of_pobjects = ' + str(Scale_of_pobjects) + '\n' +
                        'Random_orientation = ' + str(Random_orientation) + '\n' +
                        'Num_of_impurities = ' + str(Num_of_impurities) + '\n' +
                        'Contrast = ' + str(Contrast) + '\n' +
                        'Brightness = ' + str(Brightness) + '\n' +
                        'Motion_blur = ' + str(Motion_blur) + '\n' +
                        'Gaussian_blur = ' + str(Gaussian_blur)
                        )

    transparent_paths = get_image(transparent_image_path, postfix = '.png')  
    impurities_paths = get_image(impurities_path, postfix = '.png')  

    if image_path:
        img_paths, annos = get_dataset(anno_path, image_path)
        dataset_size = len(img_paths)
        if pure_background_path:
            back_image = get_image(pure_background_path, postfix = '.jpg')
            random_selection = random.sample(back_image, int(dataset_size*Ratio_of_pure_background))
            img_paths.extend(random_selection)

        for i in range(len(img_paths)):
            if i<len(annos):

                raw_bbox = Get_bbox(annos[i])
            else:
                raw_bbox = []
            #output_img = cv2.imread(img_paths[i])


            for j in range(REPEATE):
                output_img = cv2.imread(img_paths[i])
                new_annos = []
                for box in raw_bbox:
                    new_annos.append(box) 

                num__objects = random.randint(0, Num_of_pasted_objects)
                num_impurities = random.randint(0, Num_of_impurities)
                #num = 5
                #starttime = time.time()
                for i_m in range(num__objects):
                    idx = random.randint(1, len(transparent_paths)-1)
                    transparent = transparent_paths[idx]
                    transparent_image = cv2.imread(transparent, cv2.IMREAD_UNCHANGED)
                    parts = transparent.split('_')[-1]
                    class_id = int(parts[:-4])
                    output_img, object = paste_image_with_transform(output_img,transparent_image,class_id,Scale_of_pobjects,Random_orientation)

                    new_annos.append(object)

                # add impurities
                for i_im in range(num_impurities):
                    id = random.randint(1, len(impurities_paths)-1)
                    impurity = impurities_paths[id]
                    impurity_img = cv2.imread(impurity, cv2.IMREAD_UNCHANGED)
                    output_img, _ = paste_image_with_transform(output_img, impurity_img)

                
                output_img = adjust_illumination(output_img, alpha=random.randint(1,Contrast), beta=random.randint(0,Brightness))
                Gaussian_blur = random.randint(1,Gaussian_blur)
                if Gaussian_blur%2==0:
                    Gaussian_blur=Gaussian_blur-1
                output_img = cv2.GaussianBlur(output_img, ksize=(Gaussian_blur,Gaussian_blur), sigmaX=0, sigmaY=0)
                output_img = motion_blur(output_img, kernel_size=random.randint(1,Motion_blur))

                new_anno_name = AUG_DST_DIR + '/labels/' + str(i) + '_'+ str(j) +'.txt'
                new_img_name = AUG_DST_DIR + '/images/' + str(i) + '_'+ str(j) + postfix
                cv2.imwrite(new_img_name, output_img)
                print ("writing", new_img_name)
                with open(new_anno_name, 'w') as out_file:

                    for bbox in new_annos:
                        #for box in bbox:
                        out_file.write(" ".join(str(x) for x in bbox) + '\n')
                out_file.close()






def paste_image_with_transform(background, transparent_image, class_id =0,Scale_of_pobjects=(0.8,1.2),Random_orientation=True):
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


# adjust contrast and brightness
def adjust_illumination(image, alpha=1, beta=0):
    """
    Adjust the illumination of an image using alpha (contrast) and beta (brightness).

    :param image: Input image
    :param alpha: Contrast control (1.0 means no change)
    :param beta: Brightness control (0 means no change)
    :return: Adjusted image
    """
    #adjusted_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def motion_blur(image, kernel_size=25):
    # Generate a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size

    # Apply the motion blur kernel to the image
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image



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
    

    image_path = '/home/jayc/jayc/CV/Dataset/experiment/100/train/images/train'        
    anno_path = '/home/jayc/jayc/CV/Dataset/experiment/100/train/labels/train'
    transparent_image_path ='/home/jayc/jayc/CV/Dataset/experiment/100/transparent_img'
    impurities_path = '/home/jayc/jayc/CV/Dataset/experiment/100/impurities'
    AUG_DST_DIR = '/home/jayc/jayc/CV/Dataset/experiment/100/synthetic'
    pure_background_path = '/home/jayc/jayc/CV/Dataset/experiment/100/background'
    csv_file_path = "exp.csv"
    
    # 本文件根据 正交试验生成的CSV 文件进行参数设置生成合成数据集
    # image_path ： 原始图像文件夹
    # anno_path: 原始标注文件夹
    # transparent_image_path: 由crop.py 扣出的各类目标的图像数据集，。png格式
    # impurities_path: 准备粘贴到图像的杂质图像，作为噪音或者模拟不同环境
    # AUG_DST_DIR： 合成数据集保存路径
    # pure_background_path： 空白背景图路径，通过粘贴目标或杂质进行数据合成
    # csv_file_path： 正交实验生成的CSV 文件路径

    dict = {'1': (1,1), '0.6~1.6':(0.6,1.6), '0.8~1.2': (0.8,1.2)}

    # 打开CSV文件并按行读取
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        line_count=0
        # 遍历每一行数据
        for row in reader:
            if line_count==0:
                line_count = line_count+1
                continue
            line_count = line_count+1
            Ratio_of_pure_background = float(row[0])
            Dataset_magnification = int(row[1])
            Num_of_pasted_objects = int(row[2])
            Scale_of_pobjects = dict[row[3]]
            Random_orientation = bool(row[4])
            Num_of_impurities = int(row[5])
            Contrast = int(row[6])
            Brightness = int(row[7])
            Motion_blur =int(row[8])
            Gaussian_blur=int(row[9])
            exp_id = row[11]
            DST_folder = AUG_DST_DIR + '/' + exp_id

            print("Generating synthetic dataset", line_count)

            # yaml 
            yaml_data = {'train':DST_folder+'/images', 
                        'val': '/home/jayc/jayc/CV/Dataset/experiment/100/val/images/val',
                        'test': '/home/jayc/jayc/CV/Dataset/experiment/test/images/test',
                        'nc': 2,
                        'names': ['Nauplius_Live', 'Nauplius_Shell'] 
                        }
            if(not os.path.exists(AUG_DST_DIR+ '/yaml')):
                os.mkdir(AUG_DST_DIR+ '/yaml')  

            yaml_file_path = AUG_DST_DIR + '/yaml/' + exp_id + '.yaml'
            with open(yaml_file_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)

            # ## paste factors
            # Ratio_of_pure_background=1  # ratio of pure background to raw dataset
            # Dataset_magnification = 20
            # Num_of_pasted_objects = 5
            # Scale_of_pobjects=(0.8,1.3)
            # Random_orientation = True
            # Num_of_impurities=50
            # Contrast = 2     # 1 means no change
            # Brightness = 0   #0 means no change
            # Motion_blur = 3     # 1 means no change
            # Gaussian_blur=11   # 1 means no change

            #Paste_mask(image_path = IMG_DIR, anno_path = ANNO_DIR, transparent_image_path = DST_DIR, pure_background_path = None, AUG_DST_DIR = './dst', REPEATE=1, postfix = '.jpg')
            Paste_mask(image_path = image_path, 
                        anno_path = anno_path, 
                        transparent_image_path = transparent_image_path, 
                        impurities_path = impurities_path,
                        pure_background_path = pure_background_path, 
                        AUG_DST_DIR = DST_folder,
                        Ratio_of_pure_background = Ratio_of_pure_background, 
                        REPEATE = Dataset_magnification, 
                        Num_of_pasted_objects = Num_of_pasted_objects,
                        Scale_of_pobjects=Scale_of_pobjects,
                        Random_orientation = Random_orientation,
                        Num_of_impurities = Num_of_impurities,
                        Contrast = Contrast,
                        Brightness = Brightness,
                        Motion_blur = Motion_blur,
                        Gaussian_blur = Gaussian_blur,
                        postfix = '.jpg',
                        exp_id = exp_id
                        )



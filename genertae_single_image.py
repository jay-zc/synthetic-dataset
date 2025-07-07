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

postfix = '.jpg' 


def Paste_mask(image_path = None, 
                anno_path = None, 
                transparent_image_path = None, 
                impurities_path=None,
                pure_background_path = None, 
                AUG_DST_DIR = None, 
                REPEATE=1, 
                Num_of_pasted_objects=0,
                Scale_of_pobjects=1,
                Random_orientation = True,
                Num_of_impurities=100,
                Contrast = 1,
                Brightness = 0,
                Motion_blur=1,
                Gaussian_blur=1,
                postfix = '.jpg'):
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

    transparent_paths = get_image(transparent_image_path, postfix = '.png')  
    impurities_paths = get_image(impurities_path, postfix = '.png')  

    if image_path:
        img_paths, annos = get_dataset(anno_path, image_path)
        if pure_background_path:
            back_image = get_image(pure_background_path, postfix = None)
            img_paths.extend(back_image)
            img_paths = back_image
            annos = []
            

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
                    #transparent_image = cv2.resize(transparent_image, (1024, 1024))

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

                new_anno_name_instances = AUG_DST_DIR + '/labels/instances' + str(i) + '_'+ str(j) +'.txt'
                new_img_name_instances = AUG_DST_DIR + '/images/instances' + str(i) + '_'+ str(j) + postfix
                cv2.imwrite(new_img_name_instances, output_img)
                with open(new_anno_name_instances, 'w') as out_file:

                    for bbox in new_annos:
                        #for box in bbox:
                        out_file.write(" ".join(str(x) for x in bbox) + '\n')
                out_file.close()
                
                                
                output_img = adjust_illumination(output_img, alpha=random.randint(1,Contrast), beta=random.randint(0,Brightness))
                Gaussian_blur = random.randint(1,Gaussian_blur)
                if Gaussian_blur%2==0:
                    Gaussian_blur=Gaussian_blur-1
                output_img = cv2.GaussianBlur(output_img, ksize=(Gaussian_blur,Gaussian_blur), sigmaX=0, sigmaY=0)
                output_img = motion_blur(output_img, kernel_size=random.randint(1,Motion_blur))
                # change color a bit randomly



                new_anno_name = AUG_DST_DIR + '/labels/' + str(i) + '_'+ str(j) +'.txt'
                new_img_name = AUG_DST_DIR + '/images/' + str(i) + '_'+ str(j) + postfix
                cv2.imwrite(new_img_name, output_img)
                print ("writing",img_paths[i],"to->", new_img_name)
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
    #height, width = height, width if height>1 and width>1 else height+1, width+1
    height = height if height>1 else height+1
    width = width if width>1 else width+1

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
    if postfix == None:
        image_files = [f for f in os.listdir(image_folder)]
    elif postfix == '.png' or postfix == '.jpg':
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(postfix) or f.lower().endswith('.jpeg') or f.lower().endswith('.jpg') or f.lower().endswith('.tiff')]
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
    

    # image_path = '/home/jayc/jayc/CV/Dataset/experiment/100/train/images'
    # anno_path = '/home/jayc/jayc/CV/Dataset/experiment/100/train/labels'
    # transparent_image_path ='/home/jayc/jayc/CV/Dataset/experiment/100/transparent_img'
    # impurities_path = '/home/jayc/jayc/CV/Dataset/experiment/100/impurities'
    # AUG_DST_DIR = '/home/jayc/jayc/CV/Dataset/experiment/100/synthetic/1'

    image_path = r'/home/jayc/jayc/synthetic/Data/images/val'
    anno_path = r'/home/jayc/jayc/synthetic/Data/labels/val'
    transparent_image_path = '/home/jayc/jayc/synthetic/Data/transparent_checked'
    impurities_path = '/home/jayc/jayc/synthetic/Data/impurities'
    AUG_DST_DIR = 'temp'




    ## paste factors
    pure_background_path = 'background' # '/mnt2/jayc_2T/Ha/Jayc/c_training/purebackground'
    Dataset_magnification = 100
    Num_of_pasted_objects = 20
    Scale_of_pobjects=(0.8,1.3)
    Random_orientation = True
    Num_of_impurities=80
    Contrast = 3     # 1 means no change
    Brightness = 2   #0 means no change
    Motion_blur = 5     # 1 means no change
    Gaussian_blur=7   # 1 means no change

    #Paste_mask(image_path = IMG_DIR, anno_path = ANNO_DIR, transparent_image_path = DST_DIR, pure_background_path = None, AUG_DST_DIR = './dst', REPEATE=1, postfix = '.jpg')
    Paste_mask(image_path = image_path, 
                anno_path = anno_path, 
                transparent_image_path = transparent_image_path, 
                impurities_path = impurities_path,
                pure_background_path = pure_background_path, 
                AUG_DST_DIR = AUG_DST_DIR, 
                REPEATE = Dataset_magnification, 
                Num_of_pasted_objects = Num_of_pasted_objects,
                Scale_of_pobjects=Scale_of_pobjects,
                Random_orientation = Random_orientation,
                Num_of_impurities = Num_of_impurities,
                Contrast = Contrast,
                Brightness = Brightness,
                Motion_blur = Motion_blur,
                Gaussian_blur = Gaussian_blur,
                postfix = '.jpg'
                )

import cv2
import numpy as np
from matplotlib import pyplot as plt
import shutil
import os

def visualize(img,folder_path):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.axis('equal')
    plt.imshow(image)
    plt.savefig(folder_path, bbox_inches = None)
    plt.close()

def get_rgb_of_region(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_size = image.shape
    top_left_x = int(img_size[1]*0.98)
    top_left_y = int(img_size[0]*0.98)
    bottom_right_x = img_size[1]
    bottom_right_y = img_size[0]

    region1 = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] #right bottom
    average_color_per_row1 = np.average(region1, axis=0)
    average_color1 = np.average(average_color_per_row1, axis=0)

    region2 = image[0:int(img_size[0]*0.02), top_left_x:bottom_right_x] #right top
    average_color_per_row2 = np.average(region2, axis=0)
    average_color2 = np.average(average_color_per_row2, axis=0)    

    region3 = image[top_left_y:bottom_right_y, 0:int(img_size[1]*0.02)] #left bottom
    average_color_per_row3 = np.average(region3, axis=0)
    average_color3 = np.average(average_color_per_row3, axis=0)    

    region4 = image[0:int(img_size[0]*0.02), 0:int(img_size[1]*0.02)] #left top
    average_color_per_row4 = np.average(region4, axis=0)
    average_color4 = np.average(average_color_per_row4, axis=0)      
    # visualize(region,check_folder_path+filename)    
    return average_color1, average_color2, average_color3, average_color4


data_folder_path =  "C:/Users/xinl55/github/HAM10000/HAM10000/images/" #"C:/Users/xinl55/github/test/"
check_folder_path1 = "C:/Users/xinl55/github/HAM10000/true/"
check_folder_path2 = "C:/Users/xinl55/github/HAM10000/false/"
if not os.path.isdir(check_folder_path1):
    print("Create folder path:", check_folder_path1)
    os.mkdir(check_folder_path1)
if not os.path.isdir(check_folder_path2):
    print("Create folder path:", check_folder_path2)
    os.mkdir(check_folder_path2)
check_number = 0
for filename in os.listdir(data_folder_path):
    if filename.endswith(".jpg"):
        rgb_values1, rgb_values2, rgb_values3, rgb_values4 = get_rgb_of_region(data_folder_path+filename)
        found_flag = False
        for rgb_values in [rgb_values1, rgb_values2, rgb_values3, rgb_values4]:
            if np.all(rgb_values < 50):
                found_flag = True
                break
        if found_flag:
            shutil.copyfile(data_folder_path+filename, check_folder_path1+filename)
            # visualize(data_folder_path+filename, check_folder_path1+filename)
            check_number = check_number +1
        else:
            shutil.copyfile(data_folder_path+filename, check_folder_path2+filename)
            # visualize(data_folder_path+filename, check_folder_path2+filename)
print("Total number: ",check_number)
        
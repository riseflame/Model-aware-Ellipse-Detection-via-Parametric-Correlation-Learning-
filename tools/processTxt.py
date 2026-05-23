import os
import cv2
import json
import numpy as np
from PIL import Image
import random
#归一化椭圆参数
def ellipseRegularized(e):
    # e: 5 x n, 每一列是 (cx, cy, a, b, theta_rad)
    
    # theta_rad 限定在 -pi~pi 之间
    e = [float(item) for item in e]
    e[4] = np.deg2rad(e[4])  
    e[4] = -e[4]
    
    # 规则化使长轴为第三个参数，短轴为第四个参数

    if(e[3]>e[2]):
        min_axes = e[2]
        e[2] = e[3]
        e[3] = min_axes
        if(e[4] > 0):
            e[4] = e[4]  - np.pi/2
            #e[4] = e[4]  - 90
        else:
            e[4] = e[4]  + np.pi/2
            #e[4] = e[4]  + 90
    e[2] = e[2]/2
    e[3] = e[3]/2
    mid = e[1]
    e[1] = e[0]
    e[0] = mid

    
    return e

txt_list = os.listdir('/fastersharefiles/liuzezheng/Synthetic Images - Overlap Ellipses/test')
for file in txt_list:
        gt = "/fastersharefiles/liuzezheng/Synthetic Images - Overlap Ellipses/test/"  + file[:-4] + ".txt"
        processed_gt = "/fastersharefiles/liuzezheng/Synthetic Images - Overlap Ellipses/processed_test/"  + file[:-4] + ".txt"
        with open(gt, "r") as f:
            lines = f.readlines()
        num_entries = len(lines) - 1  # 第一行是标题，所以总条目数为行数减1
        with open(processed_gt, 'w') as file:
            # 写入每个文档条目的数量作为新的第一行
            file.write(str(num_entries) + '\n')
            for line in lines[1:]:
                #.strip 去掉两边的制表符
                regularized_line = ellipseRegularized(line.strip().split()[1:])
                regularized_line = [str(item) for item in regularized_line]
                line_t= [item + '\t' for item in regularized_line[0:]]
                new_line = ' '.join(line_t) + '\n'
                file.write(new_line)
            



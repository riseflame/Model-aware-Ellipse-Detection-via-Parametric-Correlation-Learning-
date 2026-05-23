import torchcam
import os
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import pycocotools.coco as coco
from backbone.dlanet_dcn import MyNet
from backbone.dlanet_dcn import DlaNet
from Loss import _gather_feat, _transpose_and_gather_feat
from dataset import get_affine_transform, get_edge
from pyheatmap.heatmap import HeatMap
import math
import measure
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp 
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F 
features_blobs = []
def pre_process(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

    inp_image = (inp_image / 255.).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 三维reshape到4维，（1，3，512，512）

    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta
#def hook_feature(module, input, output): # hook注册, 响应图提取

    #print("hook input",input[0].shape)
    #features_blobs.append(output.data.cpu().numpy())

def predict():
    device = torch.device('cuda')
    model = MyNet(34)
    model.load_state_dict(torch.load('./results/train/best.pth'))
    model = model.eval().to(device)
    data_coco = coco.COCO('../new_ged/test/annotations/test.json')
    imgs_id = data_coco.getImgIds()
    cam_extractor = SmoothGradCAMpp(model)
    for index in imgs_id:
        file_name = data_coco.loadImgs(ids=[index])[0]['file_name']
        image_name = os.path.join('../new_ged/test/images', file_name)
        ann_ids = data_coco.getAnnIds(imgIds=[index])
        anns = data_coco.loadAnns(ids=ann_ids)
        print(image_name)
        image = cv2.imread(image_name)
        img = image
        images, meta = pre_process(image)
        global  img_x,img_y
        img_x = img.shape[0]  
        img_y = img.shape[1]    
        gt = []
        for ann in anns:
            gt.append(ann['bbox'])
        images = images.to(device)
        output = model(images)
        #h_x = F.softmax(output, dim=1).data.squeeze()  
        activation_map = cam_extractor('hm', output)
        activation_map = activation_map[0].detach().cpu().numpy()
        img_pil = Image.open(image_name)
        result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.4)
        save_path = os.path.join('./heatmap', file_name.split('/')[-1])
        result.save(save_path,quality=95)
        # plt.imshow(result)
        # plt.axis('off')
if __name__ == '__main__':
    features_blobs = []
    predict()
    print(123)
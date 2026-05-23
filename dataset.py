import os
import cv2
import math
import random
import numpy as np
import torch.utils.data as data
import pycocotools.coco as coco
import torch

class ctDataset(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]

    def __init__(self, data_dir='./data', split='train'):
        self.data_dir = data_dir
        self.split = split
        try:
            if split == 'train':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train.json')
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'test.json')
        except:
            print('No any data!')

        self.max_objs = 100
        self.class_name = ['obj']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.data_dir, 'images', file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        keep_res = False  #
        if keep_res:
            input_h = (height | 31) + 1
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = 512, 512

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        # Augment
        #inp = grayscale(inp, 0.5)
        inp = get_edge(inp, prob=1)

        inp = (inp.astype(np.float32) / 255.)

        # 归一化
        inp = inp.transpose(2, 0, 1)

        down_ratio = 4
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_top = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        ab = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        #中点偏移量 大小100*2
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        mask = np.zeros((num_classes, 128, 128), dtype=np.float32)

        #顶点偏移量 索引 mask 一个椭圆上采样20个点计算偏移量 数据集中最多有50+ 个椭圆 所以定义1200*2足够大了
        t = np.linspace(0, 2*np.pi, 21)
        t = t[0:20]
        reg_top = np.zeros((self.max_objs*12, 2), dtype=np.float32)
        ind_top = np.zeros((self.max_objs*12), dtype=np.int64)
        top_mask = np.zeros((self.max_objs*12), dtype=np.uint8)
        draw_gaussian = draw_umich_gaussian
        #椭圆参数 第一位放多少个目标
        param =  torch.tensor([num_objs]) 
        #顶点循环计数器 
        count_top = 0
        for k in range(num_objs):
            
            #顶点循环计数器
            count_top += 1 
            ann = anns[k]
            bbox = ann['bbox']  # x,y,angle,a,b
            cls_id = int(self.cat_ids[ann['category_id']])

            # 数据扩充和resize之后的变换
            bbox[:2] = affine_transform(bbox[:2], trans_output)

            bbox[2:4] = affine_transform(bbox[2:4], trans_output, bbox[4], mode='ab')

            #中点坐标 限幅
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            #a b
            bbox[2] = np.clip(bbox[2], 0, output_w - 1)
            bbox[3] = np.clip(bbox[3], 0, output_h - 1)

            

            a, b, an = bbox[2], bbox[3], bbox[4]

            #计算4个顶点的坐标
            p1 = np.array([bbox[0] + a*math.cos(math.radians(an)),bbox[1] + a*math.sin(math.radians(an))]).squeeze(-1)
            p2 = np.array([bbox[0] - a*math.cos(math.radians(an)),bbox[1] - a*math.sin(math.radians(an))]).squeeze(-1)
            p3 = np.array([bbox[0] + b*math.sin(math.radians(an)),bbox[1] - b*math.cos(math.radians(an))]).squeeze(-1)
            p4 = np.array([bbox[0] - b*math.sin(math.radians(an)),bbox[1] + b*math.cos(math.radians(an))]).squeeze(-1)


            #限幅
            p1[0] = np.clip(p1[0], 0, output_w - 1)
            p1[1] = np.clip(p1[1], 0, output_h - 1)
            p2[0] = np.clip(p2[0], 0, output_w - 1)
            p2[1] = np.clip(p2[1], 0, output_h - 1)
            p3[0] = np.clip(p3[0], 0, output_w - 1)
            p3[1] = np.clip(p3[1], 0, output_h - 1)
            p4[0] = np.clip(p4[0], 0, output_w - 1)
            p4[1] = np.clip(p4[1], 0, output_h - 1)


            #椭圆上均匀采样20个点的坐标
            px = (bbox[0] + a*np.cos(t)).astype(int) 
            py = (bbox[1] + b*np.sin(t)).astype(int) 
            px = np.clip(px, 0, output_w - 1)
            py = np.clip(py, 0, output_h - 1)
            #这20个点到圆心的偏移量
            ox = bbox[0]-px
            oy = bbox[1]-py



            # #计算距离中点的偏差
            # p1_x[0] =  bbox[0]-p1[0]
            # p1_y[0] =  bbox[1]-p1[1]
            # p2_x[0] =  bbox[0]-p2[0]
            # p2_y[0] =  bbox[1]-p2[1]
            # p3_x[0] =  bbox[0]-p3[0]
            # p3_y[0] =  bbox[1]-p3[1]
            # p4_x[0] =  bbox[0]-p4[0]
            # p4_y[0] =  bbox[1]-p4[1]
            #计算距离p2,p3,p4的偏差
            # for z in range(1,4):
            #     #顶点p1对顶点p2 3 4
            #     p1_x[z] = p1[0] - P[z][0]
            #     p1_y[z] = p1[1] - P[z][1]
            #     #顶点p2对顶点p2 3 4
            #     p2_x[z] = p2[0] - P[z][0]
            #     p2_y[z] = p2[1] - P[z][1]
            #     #顶点p1对顶点p2 3 4
            #     p3_x[z] = p3[0] - P[z][0]
            #     p3_y[z] = p3[1] - P[z][1]
            #     #顶点p1对顶点p2 3 4
            #     p4_x[z] = p4[0] - P[z][0]
            #     p4_y[z] = p4[1] - P[z][1]

            
            if a > 0 and b > 0:
                radius = gaussian_radius((math.ceil(b * 2.0), math.ceil(a * 2.0)))
                radius = max(0, int(radius))
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                #顶点坐标转int
                ct_int_top1 = p1.astype(np.int32)
                ct_int_top2 = p2.astype(np.int32)
                ct_int_top3 = p3.astype(np.int32)
                ct_int_top4 = p4.astype(np.int32)
                #中点的热力图
                draw_gaussian(hm[cls_id], ct_int, radius)


                #画20个点的热力图
                #画出来的高斯分布应该不能有重合区域，否则就在边界处概率更高了
                # rad = int(min((math.sqrt(a**2+b**2))/2,b))
                #画顶点的热力图
                #画出来的高斯分布应该不能有重合区域，否则就在边界处概率更高了
                rad = int(min((math.sqrt(a**2+b**2))/2,b))
                draw_gaussian(hm_top[cls_id], ct_int_top1,rad)
                draw_gaussian(hm_top[cls_id], ct_int_top2, rad)
                draw_gaussian(hm_top[cls_id], ct_int_top3, rad)
                draw_gaussian(hm_top[cls_id], ct_int_top4, rad)



                ab[k] = 1. * a, 1. * b
                ang[k] = 1. * an + 90
                ind[k] = ct_int[1] * output_w + ct_int[0]
                #float型中点坐标-int型，通过这个reg分支回归的offset来弥补下采样精度损失
                reg[k] = ct - ct_int
                reg_mask[k] = 1


                # #顶点偏移量 顶点1 对 中点 顶点2 顶点3 顶点4 的偏移量
                # if count_top*20 == 0 :
                #     print(123)
                #print(reg_top[(count_top-1)*20:count_top*20].shape,"count:",count_top)
                # reg_top[(count_top-1)*20:count_top*20] = np.transpose([ox,oy])
                # ind_top[(count_top-1)*20:count_top*20] =  py* output_w + px
                # top_mask[(count_top-1)*20:count_top*20] = 1
               



                
                


                


                #mask分支
                cv2.ellipse(mask[cls_id], (int(ct[0]), int(ct[1])), (int(ab[k][0]), int(ab[k][1])), int(ang[k][0] - 90),
                            0, 360, 1, -1)
                
                tmp = torch.tensor([int(ct[0]), int(ct[1]), ab[k][0], ab[k][1], int(ang[k][0] - 90)])
                param = torch.cat([param,tmp])

                

        # inp: 512*512 input | hm: heatmap class | reg_mask: obj data mask | ind: center pixel index
        # wh: width & height | ang: angle
        if (501-param.shape[0]) > 0:
            rand = torch.zeros(501-param.shape[0])
            param = torch.cat([param,rand])
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'ab': ab, 'ang': ang, 'mask': mask,'hm_top':hm_top,\
               'ind_top':ind_top,'top_mask':top_mask,'param':param }
        reg_offset_flag = True  #
        if reg_offset_flag:
            ret.update({'reg': reg})
        return ret


def get_edge(img, prob=0.5):
    if random.randint(0, 9) in range(0, int(10 * prob)):
        img_b = img[:, :, 0]
        img_g = img[:, :, 1]
        img_r = img[:, :, 2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        img_1 = cv2.Laplacian(img, -1, ksize=5)
        img_1 = cv2.normalize(img_1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_2 = cv2.Canny(img, 50, 150)
        img_2 = cv2.normalize(img_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_3 = 0.5 * cv2.Sobel(img, -1, 0, 1, 5) + 0.5 * cv2.Sobel(img, -1, 1, 0, 5)
        img_3 = cv2.normalize(img_3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        img_4 = cv2.normalize(img_4, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        img = np.stack([img_b, img_g, img_r, img_1, img_2, img_3, img_4], axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return img


def grayscale(img, prob=0.5):
    if random.randint(0, 9) in range(0, int(10 * prob)):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t, angle=0, mode='xy'):
    if mode == 'xy':
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
    elif mode == 'ab':
        angle = np.deg2rad(angle)
        cosA = np.abs(np.cos(angle))
        sinA = np.abs(np.sin(angle))
        a_x = pt[0] * cosA
        a_y = pt[0] * sinA
        b_x = pt[1] * sinA
        b_y = pt[1] * cosA
        new_pt_a = np.array([a_x, a_y, 0.], dtype=np.float32).T
        new_pt_a = np.dot(t, new_pt_a)
        new_pt_b = np.array([b_x, b_y, 0.], dtype=np.float32).T
        new_pt_b = np.dot(t, new_pt_b)
        new_pt = np.zeros((2, 1), dtype=np.float32)
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)
    return new_pt[:2]


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

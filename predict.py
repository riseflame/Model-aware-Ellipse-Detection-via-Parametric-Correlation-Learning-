import argparse
import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import pycocotools.coco as coco
from backbone.dlanet_dcn import MyNet
from Loss import _gather_feat, _transpose_and_gather_feat
from dataset import get_affine_transform
import math
import torch.nn .functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Run EDP2Net/ElDet inference on a COCO-style dataset.")
    parser.add_argument("--weights", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--annotations", type=str, required=True, help="COCO annotation json")
    parser.add_argument("--image_dir", type=str, required=True, help="image directory")
    parser.add_argument("--output_dir", type=str, default="results/predict", help="visualized output directory")
    parser.add_argument("--det_dir", type=str, default="results/det", help="text detection output directory")
    parser.add_argument("--edge_dir", type=str, default=None, help="optional edge map output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--score_thresh", type=float, default=0.3, help="detection score threshold")
    parser.add_argument("--top_thresh", type=float, default=0.5, help="top-point score threshold")
    return parser.parse_args()


def build_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def write2txt(res, file_name, det_dir):
    os.makedirs(det_dir, exist_ok=True)
    det_path = os.path.join(det_dir, "det_" + os.path.splitext(os.path.basename(file_name))[0] + ".bmp.txt")
    with open(det_path, "w") as wf:
        wf.write("{}\n".format(len(res)))
        for class_name, x, y, a, b, ang, prob in res:
            wf.write("{} {} {} {} {}\n".format(x, y, a, b, ang))

def draw(img, result,top):
    '''Draw ellipse box'''
    for class_name, x, y, a, b, ang, prob in result:
        print('pred:', [x, y, a, b, ang])
        result = np.array(result)
        x = int(x)
        y = int(y)
        a = int(a)
        b = int(b)
        angle = int(ang)
        # cv2.ellipse(img, (x, y), (a, b), angle, 0, 360, (255, 255, 0), 3)
        cv2.ellipse(img, (x, y), (a, b), angle, 0, 360, (0, 0, 255), 3)
        #鐢诲渾蹇?
        #cv2.circle(img,(x,y),3,(0, 255, 255),2)

    #鐢婚《鐐?
    #for i in range(0,np.array(top).shape[0]):
        #cv2.circle(img,(int(top[i][0]),int(top[i][1])),6,(0, 0, 255),-1)
    return img

#鐢籊T鐨勫嚱鏁?
def drawGT(img, gt):
    for  x, y, a, b, ang in gt:
        x = int(x)

        y = int(y)
        a = int(a)
        b = int(b)
        #
        angle = int(ang)
        # cv2.ellipse(img, (x, y), (a, b), angle, 0, 360, (255, 255, 0), 3)
        cv2.ellipse(img, (x, y), (a, b), angle, 0, 360, (255, 255, 0), 3)
        #鐢诲渾蹇?
        #cv2.circle(img,(x,y),3,(255,255, 0),1)
        #璁＄畻4涓《鐐圭殑鍧愭爣
        p1 = np.array([x + a*math.cos(math.radians(angle)),y + a*math.sin(math.radians(angle))])
        p2 = np.array([x - a*math.cos(math.radians(angle)),y - a*math.sin(math.radians(angle))])
        p3 = np.array([x + b*math.sin(math.radians(angle)),y - b*math.cos(math.radians(angle))])
        p4 = np.array([x - b*math.sin(math.radians(angle)),y + b*math.cos(math.radians(angle))])
        #闄愬箙
        p1[0] = np.clip(p1[0], 0, np.array(img).shape[1] - 1)
        p1[1] = np.clip(p1[1], 0, np.array(img).shape[0] - 1)
        p2[0] = np.clip(p2[0], 0, np.array(img).shape[1] - 1)
        p2[1] = np.clip(p2[1], 0, np.array(img).shape[0] - 1)
        p3[0] = np.clip(p3[0], 0, np.array(img).shape[1] - 1)
        p3[1] = np.clip(p3[1], 0, np.array(img).shape[0] - 1)
        p4[0] = np.clip(p4[0], 0, np.array(img).shape[1] - 1)
        p4[1] = np.clip(p4[1], 0, np.array(img).shape[0] - 1)
        

       
        #cv2.circle(img,(int(p1[0]),int(p1[1])),5,(255, 255, 0),3)
        #cv2.circle(img,(int(p2[0]),int(p2[1])),5,(255, 255, 0),3)
        #cv2.circle(img,(int(p3[0]),int(p3[1])),5,(255, 255, 0),3)
        #cv2.circle(img,(int(p4[0]),int(p4[1])),5,(255, 255, 0),3)


    return img

#鐢籬eat map鐨勫嚱鏁?
def drawHM(img, hm):
    #hm 1 1 128 128
    #hm = hm.sigmoid().cpu()
    hm = hm.cpu()
    #hm_up = cv2.resize(hm[0,0,:,:].detach().numpy(), (img_y,img_x), interpolation=cv2.INTER_AREA) 
    hm_up = F.upsample(hm, size=[img_x, img_y], mode='bilinear', align_corners=False)
    alpha = 0.6 # 璁剧疆瑕嗙洊鍥剧墖鐨勯€忔槑搴?
    #hm_up = np.expand_dims(hm_up,-1)
    hm_up = torch.squeeze(hm_up,0)
    hm_up = torch.squeeze(hm_up,0).detach().numpy()
    hm_up = np.expand_dims(hm_up,-1)
    heat_map = np.concatenate((hm_up,hm_up,hm_up),axis=-1)
    #heat_map = 255*((heat_map - heat_map.min()) / (heat_map.max() - heat_map.min() + 1e-8)).astype(np.uint8)
    heat_map = (heat_map*255).astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_HOT)
    overlay = img.copy()
    #cv2.rectangle(overlay, (0, 0), (img_y, img_x), (255, 0, 0), -1) # 璁剧疆钃濊壊涓虹儹搴﹀浘鍩烘湰鑹茶摑鑹?
    #img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0) # 灏嗚儗鏅儹搴﹀浘瑕嗙洊鍒板師鍥?
    img = cv2.addWeighted(heat_map, alpha, img, 1, 0) # 灏嗙儹搴﹀浘瑕嗙洊鍒板師鍥?
    return img
 
def pre_process(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

    inp_image = (inp_image / 255.).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 涓夌淮reshape鍒?缁达紝锛?锛?锛?12锛?12锛?

    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, ab, ang, reg=None, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    #heat = _nms(heat)
    heat = _nms(heat,kernel=5)
    #heatmap 涓婃渶澶х殑100涓偣
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    #搴旇鏄姞offset
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    ab = _transpose_and_gather_feat(ab, inds)
    ab = ab.view(batch, K, 2)

    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, 1)


    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs,
                        ys,
                        ab[..., 0:1],
                        ab[..., 1:2],
                        ang - 90], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


#椤剁偣decode鍑芥暟
def cttop_decode(heat,reg=None,K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    #heatmap 涓婃渶澶х殑100涓偣
    scores, inds, clses, ys, xs = _topk(heat, K=K)


    xs = xs.view(batch, K, 1) 
    ys = ys.view(batch, K, 1) 



    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs,ys] ,dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections






def process(output):
    
    with torch.no_grad():
        global heat_map
        global hm_topCenter

        ang = output['ang']
        ab = output['ab']
        reg = output['reg']

        #regtop = output['reg_top'][:,:2,:,:]

        #椤剁偣鐨勭儹鍔涘浘
        #鍙鍖栧緱鍔犱笂
        #top_heat = _nms(output['hm_top'])
        hmtop = output['hm_top'].sigmoid_()

        # #鍒濆鍖栭《鐐?涓偣鐑姏鍥?
        # hm_topCenter = torch.zeros(hmtop.shape[0],1,hmtop.shape[2],hmtop.shape[3],device="cuda")
        # hm_topCenter = hm_topCenter - 10
        # for i in range(top_heat.shape[0]) :
        #     #姣忓紶鍥鹃€?00涓《鐐?
        #     # 鑾峰緱 tensor 涓渶澶х殑 100 涓厓绱犲強鍏朵綅缃?
        #     k=100
        #     values, indices = torch.topk(top_heat[i,:,:,:].flatten(), k)
        #     # mask_nms = torch.gt(top_heat, 1)  # 杩樻湭缁忚繃sigmoid锛屾墍浠ュ拰0姣旓紝浠ｈ〃缃俊搴﹀ぇ浜?.5鐨勪腑鐐规墠琚噰鏍?
        #     # mask_nms = mask_nms.squeeze().squeeze().detach().cpu().numpy()
        #     # idx = np.where(mask_nms)
        #     rows = indices//top_heat.shape[2] 
        #     cols = indices % top_heat.shape[3]
            # rows = idx[0]
            # cols = idx[1]
            # for j in range(k) :
            # for j in range(rows.shape[0]):
            # 娴嬭瘎鍒?  
                # if(top_heat[i][0][rows[j]][cols[j]])>0 :
                #     new_row = int(rows[j]+regtop[i][1][rows[j]][cols[j]])
                #     new_col = int(cols[j]+regtop[i][0][rows[j]][cols[j]])
                #     # new_row = int(rows[j]+regtop[i][1][rows[j]][cols[j]])
                #     # new_col = int(cols[j]+regtop[i][0][rows[j]][cols[j]])
                #     #new_row = int(rows[j]+regtop[i][1][idx[0][j]][idx[1][j]])
                #     #new_col = int(cols[j]+regtop[i][0][idx[1][j]][idx[0][j]])

                #     #闄愬箙
                #     if new_col >= 128 :
                #         new_col =127
                #     if new_row >= 128 :
                #         new_row =127
                #     if new_col < 0 :
                #         new_col =0
                #     if new_row < 0 :
                #         new_row =0
                #     hm_topCenter[i][0][new_row][new_col] = top_heat[i][0][rows[j]][cols[j]]
                #     offset = (regtop[i][1][rows[j]][cols[j]]**2 + regtop[i][0][rows[j]][cols[j]]**2).cpu()
                #     new_a = np.sqrt(offset).cuda()
                    #ab[i][0][new_row][new_col] = max(ab[i][0][new_row][new_col],new_a)
                    #ab[i][1][new_row][new_col] = min(ab[i][1][new_row][new_col],new_a)
        #鎶婄疆淇″害鍔犱笂鍘?
        #hm +=  hm_topCenter
        
        #hm = (output['hm']+hm_topCenter).sigmoid_()
        hm = (output['hm']).sigmoid_()
        #heat_map = hm.clone()


        if torch.cuda.is_available() and hm.is_cuda:
            torch.cuda.synchronize()
        #1*100*7 鐩稿綋浜庡墠100涓瘎鍒嗙殑妞渾鐨剎 y(涓偣宸茬粡鍔犱笂offset浜? a b  ang scroe classes  閲岄潰浼氬仛nms 搴旇鍦╪ms涔嬪墠鎶婇《鐐圭殑姒傜巼鐩栦笂鍘?
        dets = ctdet_decode(hm, ab, ang, reg=reg, K=100)  # K 鏄渶澶氫繚鐣欏嚑涓洰鏍?
        #涔熻繑鍥炲墠100璇勫垎鐨勬き鍦?搴旇鏄?*100*3 x y scroe
        #宸茬粡鍋歯ms浜?
        dets_top = cttop_decode(hmtop,K=100)
        #dets_top = []
        return dets,dets_top
        #return dets


def affine_transform(pt, t, angle=0, mode='xy'):
    new_pt = np.zeros(2, dtype=np.float32)
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
        new_pt = np.zeros(2, dtype=np.float32)
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size, ang, mode='xy'):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        if mode == 'ab':
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans, ang[p], 'ab')
        else:
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans, ang[p])
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h), dets[i, :, 4])
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h), dets[i, :, 4], mode='ab')

        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


#璇ュ嚱鏁扮敤鏉ュ皢heatmap涓婄殑鍧愭爣杞崲涓哄師鍥句笂鐨勫潗鏍?
def cttop_post_process(dets, c, s, h, w):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    ret = []
    for i in range(dets.shape[0]):
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h),np.zeros(100))
        ret.append((dets[i, :, :3]).tolist())
    return ret
 
def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = 1
    dets = ctdet_post_process(dets.copy(),
                              [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
        dets[0][j][:, :5] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
        kth = len(scores) - max_obj_per_img
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 2 + 1):
            keep_inds = (detections[j][:, 5] >= thresh)
            detections[j] = detections[j][keep_inds]
    return detections


def predict():
    total_time = 0
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.det_dir, exist_ok=True)
    if args.edge_dir:
        os.makedirs(args.edge_dir, exist_ok=True)

    model = MyNet(34)
    device = build_device(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    model.to(device)
    data_coco = coco.COCO(args.annotations)
    imgs_id = data_coco.getImgIds()
    times = []
    
    for index in imgs_id:
        file_name = data_coco.loadImgs(ids=[index])[0]['file_name']
        image_name = os.path.join(args.image_dir, file_name)
        image = cv2.imread(image_name)
        img = image
        images, meta = pre_process(image)
       
        global  img_x,img_y
        img_x = img.shape[0]  
        img_y = img.shape[1] 

        start_time = time.time()
        file_name = data_coco.loadImgs(ids=[index])[0]['file_name']
        image = cv2.imread(image_name)
        
        
        images, meta = pre_process(image)
        images = images.to(device)
        output = model(images)
        
        
        #瀵硅緭鍑虹殑tensor杩涜鍚庡鐞?杈撳嚭1锛坆atch size锛?100(top100涓娴?*7(x y a b ang off scroe)
        dets,dets_top = process(output)
        dets = post_process(dets, meta)
        end_time = time.time()
        #dets = process(output)
        #灏嗘ā鍨嬭緭鍑虹殑妞渾鍙傛暟杞崲涓烘爣鍑嗗舰寮?
        
        
        execution_time = end_time - start_time
        times.append(execution_time)

        #寰楀埌椤剁偣鐨勫潗鏍?x y scroe
        dets_top = cttop_post_process(dets_top, [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])
        fin_top = []
        for i in range(0,100) :
        #鏄剧ず椤剁偣缃俊搴﹀ぇ浜?.5鐨勭粨鏋?
            if (dets_top[0][i][2] > args.top_thresh):
                fin_top.append(dets_top[0][i])

        ret = merge_outputs(dets)

        res = np.empty([1, 7])
        for i, c in ret.items():
            #閫夋嫨鎵€鏈夌疆淇″害澶т簬0.3鐨勯娴嬬粨鏋?杩欓噷浣跨敤鐨勭疆淇″害灏辨槸妫€娴嬩腑鐐圭殑缃俊搴?
            tmp_s = ret[i][ret[i][:, 5] > args.score_thresh]
            tmp_c = np.ones(len(tmp_s)) * (i + 1)
            tmp = np.c_[tmp_c, tmp_s]
            res = np.append(res, tmp, axis=0)

        res = np.delete(res, 0, 0)
        res = res.tolist()
        #鐢婚娴嬬粨鏋?
        img = draw(img, res,fin_top)
        #鐢籫t
        #img = drawGT(img, gt)
        #鐢籬eat map
        #edge.torch.tensor
        #img = drawHM(img,1-output['edge'])
        #鐢昏竟缂樺浘
        HH, WW = image.shape[0], image.shape[1]
        contour = output['edge']
        contour = F.upsample(contour, size=[HH, WW], mode='bilinear', align_corners=False)
        #contour = contour.sigmoid().data.cpu().numpy().squeeze()
        #contour = torch.clamp(contour,0,0.8)
        contour = contour.data.cpu().numpy().squeeze()
        #contour = 255*(1-(contour - contour.min()) / (contour.max() - contour.min() + 1e-8))
        #contour = 255*((contour - contour.min()) / (contour.max() - contour.min() + 1e-8))
        contour = 255*contour 
        if args.edge_dir:
            contour_path = os.path.join(args.edge_dir, os.path.basename(file_name))
            cv2.imwrite(contour_path, contour)
        #img = drawHM(img,output['hm'])
        write2txt(res, file_name, args.det_dir)


        ann_ids = data_coco.getAnnIds(imgIds=[index])
        anns = data_coco.loadAnns(ids=ann_ids)

        # for ann in anns:
        #     bbox = ann['bbox']
        #
        #     print('gt:', (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), bbox[4])
        #     cv2.ellipse(img, (int(bbox[0]), int(bbox[1])),
        #                 (int(bbox[2]), int(bbox[3])),
        #                 int(bbox[4]), 0, 360, (0, 255, 0), 3)

        #     # cv2.ellipse(img, (int(bbox[0]), int(bbox[1])),
        #     #             (int(bbox[3]), int(bbox[4])),
        #     #             int(bbox[2]), 0, 360, (0, 0, 255), 3)   # FDDB

        save_path = os.path.join(args.output_dir, os.path.basename(file_name))
        cv2.imwrite(save_path, img)
    total_time = sum(times)
    print("processed {} images in {:.3f}s".format(len(times), total_time))
        


def predict_once(input_path, file_name, model, device=torch.device("cuda"), thresh=0.9):
    img = cv2.imread(os.path.join(input_path, file_name))
    images, meta = pre_process(img)

    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    dets = process(output)

    dets = post_process(dets, meta)
    ret = merge_outputs(dets)

    res = np.empty([1, 7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 5] > thresh]
        tmp_c = np.ones(len(tmp_s)) * (i + 1)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()

    if len(res) != 0:
        img = draw(img, res)
        save_path = os.path.join('./results/predict', file_name.split()[-1])
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    predict()

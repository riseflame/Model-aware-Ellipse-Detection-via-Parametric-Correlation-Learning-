import argparse
import os

import cv2
import numpy as np
import pycocotools.coco as coco
import torch

from backbone.dlanet_dcn import MyNet
from predict import merge_outputs, post_process, pre_process, process


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Model-aware ellipse detection via parametric correlation learning."
    )
    parser.add_argument("--weights", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--annotations", type=str, required=True, help="COCO annotation json")
    parser.add_argument("--image_dir", type=str, required=True, help="image directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--score_thresh", type=float, default=0.3, help="detection score threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="ellipse IoU threshold")
    parser.add_argument("--theta_thresh", type=float, default=10, help="angle error threshold in degrees")
    return parser.parse_args()


def build_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        return int_area * 1.0 / (area1 + area2 - int_area)
    return 0


def iou_ellipse(bbox1, bbox2, shape):
    board1 = np.zeros((shape[0], shape[1]), np.uint8)
    cv2.ellipse(
        board1,
        (int(bbox1[0]), int(bbox1[1])),
        (int(bbox1[2]), int(bbox1[3])),
        int(bbox1[4]),
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    board2 = np.zeros((shape[0], shape[1]), np.uint8)
    cv2.ellipse(
        board2,
        (int(bbox2[0]), int(bbox2[1])),
        (int(bbox2[2]), int(bbox2[3])),
        int(bbox2[4]),
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    board = board1 + board2
    inter = len(board[np.where(board > 1)])
    union = len(board[np.where(board > 0)])
    return 1.0 * inter / union if union else 0


def get_pre_ret(img_path, model, device, score_thresh=0.3):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError("Failed to read image: {}".format(img_path))

    images, meta = pre_process(image)
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
        dets, _ = process(output)

    dets = post_process(dets, meta)
    ret = merge_outputs(dets)

    res = np.empty([1, 7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 5] > score_thresh]
        tmp_c = np.ones(len(tmp_s)) * (i + 1)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    return res.tolist(), image.shape


def get_ap(img_path, gt, model, device, iou_thres=0.5, score_thresh=0.3):
    tp = []
    gt_close = []
    pre_ret, shape = get_pre_ret(img_path, model, device, score_thresh=score_thresh)

    for class_name, x, y, a, b, ang, prob in pre_ret:
        pre_one = np.array([x, y, a, b, ang])
        flag = False
        for lab in gt:
            x_l, y_l, a_l, b_l, ang_l = lab
            lab_one = np.array([x_l, y_l, a_l, b_l, ang_l])
            iou = iou_ellipse(pre_one, lab_one, shape)
            if lab not in gt_close and iou > iou_thres:
                gt_close.append(lab)
                tp.append([1, prob])
                flag = True
                break
        if not flag:
            tp.append([-1, prob])

    return tp, len(gt)


def in_range(x):
    while x > 90:
        x -= 180
    while x < -90:
        x += 180
    return x


def get_ap_theta(img_path, gt, model, device, iou_thres=0.5, theta_thres=10, score_thresh=0.3):
    tp = []
    gt_close = []
    pre_ret, shape = get_pre_ret(img_path, model, device, score_thresh=score_thresh)

    for class_name, x, y, a, b, ang, prob in pre_ret:
        pre_one = np.array([x, y, a, b, ang])
        flag = False
        for lab in gt:
            x_l, y_l, a_l, b_l, ang_l = lab
            lab_one = np.array([x_l, y_l, a_l, b_l, ang_l])
            iou = iou_ellipse(pre_one, lab_one, shape)
            angle_ok = abs(in_range(lab_one[-1] - pre_one[-1])) < theta_thres
            nearly_circle = pre_one[3] != 0 and pre_one[2] / pre_one[3] < 1.2
            if lab not in gt_close and iou > iou_thres and (angle_ok or nearly_circle):
                gt_close.append(lab)
                tp.append([1, prob])
                flag = True
                break
        if not flag:
            tp.append([-1, prob])

    return tp, len(gt)


def average_precision(tps, gt_num):
    if gt_num == 0:
        return 0.0

    tps.sort(key=lambda x: x[-1], reverse=True)
    recall, precision = [0], [0]
    true_positive, seen = 0, 0
    for tp in tps:
        seen += 1
        if tp[0] == 1:
            true_positive += 1
            recall.append(1.0 * true_positive / gt_num)
        else:
            recall.append(recall[-1])
        precision.append(1.0 * true_positive / seen)

    curve = {0: 0}
    for i in range(len(recall)):
        if recall[i] not in curve or curve[recall[i]] < precision[i]:
            curve[recall[i]] = precision[i]

    keys = sorted(curve)
    ap = 0.0
    for i in range(1, len(keys)):
        ap += curve[keys[i]] * (keys[i] - keys[i - 1])
    return ap


def evaluation(model, device, annotations, image_dir, iou_thresh=0.5, theta_thresh=10, score_thresh=0.3):
    model.eval()
    model.to(device)

    data_coco = coco.COCO(annotations)
    imgs_id = data_coco.getImgIds()

    gt_num = 0
    tps = []
    theta_gt_num = 0
    theta_tps = []
    for index in imgs_id:
        file_name = data_coco.loadImgs(ids=[index])[0]["file_name"]
        image_name = os.path.join(image_dir, file_name)
        ann_ids = data_coco.getAnnIds(imgIds=[index])
        anns = data_coco.loadAnns(ids=ann_ids)
        gt = [ann["bbox"] for ann in anns]

        tp, gt_n = get_ap(image_name, gt, model, device, iou_thresh, score_thresh)
        gt_num += gt_n
        tps.extend(tp)

        tp_theta, gt_theta_n = get_ap_theta(image_name, gt, model, device, iou_thresh, theta_thresh, score_thresh)
        theta_gt_num += gt_theta_n
        theta_tps.extend(tp_theta)

    iou_ap = average_precision(tps, gt_num)
    theta_ap = average_precision(theta_tps, theta_gt_num)
    print("* iou_ap: ", iou_ap)
    print("* theta_ap: ", theta_ap)
    return iou_ap, theta_ap


def main():
    args = parse_args()
    device = build_device(args.device)
    model = MyNet(34)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    evaluation(
        model,
        device,
        args.annotations,
        args.image_dir,
        iou_thresh=args.iou_thresh,
        theta_thresh=args.theta_thresh,
        score_thresh=args.score_thresh,
    )


if __name__ == "__main__":
    main()

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate normalized ellipse txt detections.")
    parser.add_argument("--gt_dir", type=str, required=True, help="ground-truth txt directory")
    parser.add_argument("--det_dir", type=str, required=True, help="raw detection txt directory")
    parser.add_argument("--det_out_dir", type=str, default="results/test", help="normalized detection output directory")
    parser.add_argument("--dataset", type=str, default="ged", choices=["ged", "smart", "pcb", "overlap", "occluded"])
    parser.add_argument("--split", type=str, default=" ", help="raw detection separator")
    parser.add_argument("--angle_format", type=str, default="deg", choices=["deg", "rad"])
    parser.add_argument("--beta", type=float, default=0.8, help="overlap threshold")
    return parser.parse_args()


def ellipseRegularized(e):
    if e[4] > np.pi:
        e[4] = e[4] - 2 * np.pi
    if e[4] < -np.pi:
        e[4] = e[4] + 2 * np.pi

    if e[3] > e[2]:
        min_axes = e[2]
        e[2] = e[3]
        e[3] = min_axes
        if e[4] > 0:
            e[4] = e[4] - np.pi / 2
        else:
            e[4] = e[4] + np.pi / 2

    if e[4] > np.pi:
        e[4] = e[4] - 2 * np.pi
    if e[4] < -np.pi:
        e[4] = e[4] + 2 * np.pi

    return e


def check_file_exists(directory, image_name):
    return any(file == image_name for file in os.listdir(directory))


def normal_det(input_path, output_path, split=" ", angle_format="deg"):
    os.makedirs(output_path, exist_ok=True)
    files = os.listdir(input_path)

    for filename in files:
        in_path = os.path.join(input_path, filename)
        out_path = os.path.join(output_path, filename)
        with open(in_path, "r") as f:
            data = f.readlines()
            ellipses = np.zeros((int(data[0]), 5))
            for i in range(1, int(data[0]) + 1):
                dic = data[i].split(split)
                an = dic[4].split("\n")
                ellipse = [float(dic[0]), float(dic[1]), float(dic[2]), float(dic[3]), float(an[0])]
                if angle_format == "deg":
                    ellipse[4] = np.deg2rad(ellipse[4])
                ellipse = ellipseRegularized(ellipse)
                ellipses[i - 1, 0:5] = ellipse[0:5]

        with open(out_path, "w") as f:
            f.write(data[0])
            for i in range(int(data[0])):
                line = "\t".join(str(x) for x in ellipses[i])
                f.write(line + "\n")


def check_overlap1(ellipse_param1, ellipse_param2, size_im):
    pixels_x, pixels_y = np.meshgrid(np.arange(size_im[0]) + 1, np.arange(size_im[1]) + 1)
    a1, b1, x1, y1, theta1 = ellipse_param1
    if (a1 != 0) | (b1 != 0):
        f1 = (
            ((pixels_x - x1) * np.sin(theta1) - (pixels_y - y1) * np.cos(theta1)) ** 2 / b1**2
            + ((pixels_x - x1) * np.cos(theta1) + (pixels_y - y1) * np.sin(theta1)) ** 2 / a1**2
            - 1
        )
        pixels_inside_ellipse1 = ~(f1 > 0)
    else:
        return 0

    a2, b2, x2, y2, theta2 = ellipse_param2
    if (a2 != 0) | (b2 != 0):
        f2 = (
            ((pixels_x - x2) * np.sin(theta2) - (pixels_y - y2) * np.cos(theta2)) ** 2 / b2**2
            + ((pixels_x - x2) * np.cos(theta2) + (pixels_y - y2) * np.sin(theta2)) ** 2 / a2**2
            - 1
        )
        pixels_inside_ellipse2 = ~(f2 > 0)
    else:
        return 0

    overlap_ratio = 1 - np.sum(np.logical_xor(pixels_inside_ellipse1, pixels_inside_ellipse2)) / np.sum(
        np.logical_or(pixels_inside_ellipse1, pixels_inside_ellipse2)
    )
    return overlap_ratio


def getFValue(tp, fn, fp):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_value = 2 * precision * recall / (precision + recall)
    return f_value, precision, recall


def detection_file_for_gt(gt_file, dataset, det_out_dir):
    if dataset in ("smart", "ged"):
        det_file = "det_" + gt_file[3:-8] + ".bmp.txt"
    elif dataset == "pcb":
        det_file = "det_" + gt_file[0:-4] + ".bmp.txt"
    elif dataset in ("overlap", "occluded"):
        det_file = "det_" + gt_file[0:-13] + ".bmp.txt"
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset))
    return os.path.join(det_out_dir, det_file)


def computePerformanceAllLu(path_gt, path_det_out, dataset="ged", beta=0.8):
    files = os.listdir(path_gt)
    n_files = len(files)
    tp = np.zeros((1, n_files))
    fp = np.zeros((1, n_files))
    fn = np.zeros((1, n_files))

    for i, filename in enumerate(files):
        file_gt = os.path.join(path_gt, filename)
        file_det = detection_file_for_gt(filename, dataset, path_det_out)
        if not os.path.exists(file_det):
            raise FileNotFoundError("Detection file not found: {}".format(file_det))

        with open(file_gt, "r") as f:
            data = f.readlines()
            gt_ellipses = np.zeros((int(data[0]), 5))
            for ind1 in range(1, int(data[0]) + 1):
                dic = data[ind1].split("\t")
                an = dic[4].split("\n")
                cx = float(dic[0])
                cy = float(dic[1])
                a = float(dic[2])
                b = float(dic[3])
                theta = float(an[0])
                gt_ellipses[ind1 - 1, 0:5] = [a, b, cx, cy, theta]

        gt_ellipses = np.transpose(gt_ellipses)
        sorted_indices = np.argsort(gt_ellipses[2])
        gt_ellipses = gt_ellipses[:, sorted_indices]

        with open(file_det, "r") as f:
            data = f.readlines()
            det_ellipses = np.zeros((int(data[0]), 5))
            for ind2 in range(1, int(data[0]) + 1):
                dic = data[ind2].split("\t")
                an = dic[4].split("\n")
                cx = float(dic[0])
                cy = float(dic[1])
                a = float(dic[2])
                b = float(dic[3])
                theta = float(an[0])
                det_ellipses[ind2 - 1, 0:5] = [a, b, cx, cy, theta]

        det_ellipses = np.transpose(det_ellipses)
        sorted_indices = np.argsort(det_ellipses[2])
        if len(det_ellipses.shape) > 1:
            det_ellipses = det_ellipses[:, sorted_indices]

        if (len(det_ellipses.shape) <= 1) | (len(gt_ellipses.shape) <= 1):
            tp[0, i] = 0
            fn[0, i] = np.shape(gt_ellipses)[1] - tp[0, i]
            fp[0, i] = 0
        else:
            overlap = np.zeros((gt_ellipses.shape[1], det_ellipses.shape[1]))

            for ii in range(gt_ellipses.shape[1]):
                for jj in range(det_ellipses.shape[1]):
                    max_x = max(
                        gt_ellipses[2, ii] + gt_ellipses[0, ii],
                        det_ellipses[2, jj] + det_ellipses[0, jj],
                    )
                    max_y = max(
                        gt_ellipses[3, ii] + gt_ellipses[0, ii],
                        det_ellipses[3, jj] + det_ellipses[0, jj],
                    )
                    overlap[ii, jj] = check_overlap1(
                        gt_ellipses[:, ii],
                        det_ellipses[:, jj],
                        [max_x + 5, max_y + 5],
                    )

            matched = np.count_nonzero(np.sum(overlap > beta, axis=1) > 0)
            tp[0, i] = matched
            fn[0, i] = np.shape(gt_ellipses)[1] - tp[0, i]
            fp[0, i] = det_ellipses.shape[1] - matched

    tps = np.sum(tp)
    fps = np.sum(fp)
    fns = np.sum(fn)

    if tps == 0:
        precision = 0
        recall = 0
        result_fm = 0
    else:
        precision = tps / (tps + fps)
        recall = tps / (tps + fns)
        result_fm = 2 * precision * recall / (precision + recall)
    return precision, recall, result_fm


def main():
    args = parse_args()
    normal_det(args.det_dir, args.det_out_dir, split=args.split, angle_format=args.angle_format)
    precision, recall, result_fm = computePerformanceAllLu(
        args.gt_dir,
        args.det_out_dir,
        dataset=args.dataset,
        beta=args.beta,
    )
    print(precision, recall, result_fm)


if __name__ == "__main__":
    main()

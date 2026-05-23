import argparse
import json
import os

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ellipse txt annotations to COCO-style json.")
    parser.add_argument("--image_dir", required=True, help="directory containing images")
    parser.add_argument("--txt_dir", required=True, help="directory containing ellipse txt annotations")
    parser.add_argument("--output", required=True, help="output json path")
    return parser.parse_args()


def auto_increment_integer_generator():
    i = 1
    while True:
        yield i
        i += 1


def normalize_angle(theta):
    while theta > 90 or theta < -90:
        if theta > 90:
            theta -= 180
        if theta < -90:
            theta += 180
    return theta


def convert(image_dir, txt_dir, output):
    image_id_generator = auto_increment_integer_generator()
    annotation_id_generator = auto_increment_integer_generator()
    images_list = []
    annotation_list = []

    for file in sorted(os.listdir(txt_dir)):
        if not file.endswith(".txt"):
            continue

        image_name = file[:-4] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        with Image.open(image_path) as pil:
            width, height = pil.size

        images_list.append(
            {
                "license": 1,
                "file_name": image_name,
                "height": height,
                "width": width,
                "id": next(image_id_generator),
            }
        )

        gt_path = os.path.join(txt_dir, file)
        with open(gt_path, "r") as f:
            data = f.readlines()
            for i in range(1, int(data[0]) + 1):
                dic = data[i].split("\t")
                an = dic[4].split("\n")
                cx = float(dic[0])
                cy = float(dic[1])
                a = float(dic[2])
                b = float(dic[3])
                theta = np.rad2deg(float(an[0]))

                if a < b:
                    a, b = b, a
                    theta += 90
                theta = normalize_angle(theta)

                annotation_list.append(
                    {
                        "iscrowd": 0,
                        "image_id": images_list[-1]["id"],
                        "bbox": [cx, cy, a, b, theta],
                        "category_id": 1,
                        "id": next(annotation_id_generator),
                    }
                )

    original = {"images": images_list, "annotations": annotation_list}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(json.dumps(original, indent=4))


if __name__ == "__main__":
    args = parse_args()
    convert(args.image_dir, args.txt_dir, args.output)

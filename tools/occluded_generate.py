import argparse
import os
import random

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic occluded ellipse images.")
    parser.add_argument("--output_dir", required=True, help="base output directory")
    parser.add_argument("--num_images", type=int, default=120)
    parser.add_argument("--size", type=int, default=300)
    return parser.parse_args()


def generate_ellipses_image(size, num_ellipses, txt_path):
    img100 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    img80 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    img60 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    img40 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

    with open(txt_path, "w") as file:
        file.write(str(num_ellipses) + "\n")
        for _ in range(num_ellipses):
            a = random.randint(10, int(300 / np.sqrt(2)))
            ratio = random.randint(10, 50)
            b = int(a / ratio * 10)
            x = random.randint(0, 300)
            y = random.randint(0, 300)
            angle = random.randint(0, 360)
            rad = np.deg2rad(angle)
            cv2.ellipse(img100, (x, y), (a // 2, b // 2), angle, 0, 360, (0, 0, 0), 3)
            cv2.ellipse(img80, (x, y), (a // 2, b // 2), angle, 0, int(360 * 0.80), (0, 0, 0), 3)
            cv2.ellipse(img60, (x, y), (a // 2, b // 2), angle, 0, int(360 * 0.60), (0, 0, 0), 3)
            cv2.ellipse(img40, (x, y), (a // 2, b // 2), angle, 0, int(360 * 0.40), (0, 0, 0), 3)
            file.write("{}\t{}\t{}\t{}\t{}\n".format(x, y, a // 2, b // 2, rad))

    return img100, img80, img60, img40


def main():
    args = parse_args()
    output_dirs = {
        "img100": os.path.join(args.output_dir, "img100"),
        "img80": os.path.join(args.output_dir, "img80"),
        "img60": os.path.join(args.output_dir, "img60"),
        "img40": os.path.join(args.output_dir, "img40"),
        "gt": os.path.join(args.output_dir, "gt"),
    }
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)

    img_size = (args.size, args.size)
    for i in range(args.num_images):
        num_ellipses = i % 5 * 4 + 4
        imgname = "synth_{}ellipses_img{}.jpg".format(num_ellipses, i)
        txt_path = os.path.join(output_dirs["gt"], imgname[:-4] + ".txt")
        img100, img80, img60, img40 = generate_ellipses_image(img_size, num_ellipses, txt_path)
        cv2.imwrite(os.path.join(output_dirs["img100"], imgname), img100)
        cv2.imwrite(os.path.join(output_dirs["img80"], imgname), img80)
        cv2.imwrite(os.path.join(output_dirs["img60"], imgname), img60)
        cv2.imwrite(os.path.join(output_dirs["img40"], imgname), img40)


if __name__ == "__main__":
    main()

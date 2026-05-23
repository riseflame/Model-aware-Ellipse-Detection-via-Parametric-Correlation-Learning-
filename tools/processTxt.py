import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize raw ellipse txt annotations.")
    parser.add_argument("--input_dir", required=True, help="directory containing raw txt files")
    parser.add_argument("--output_dir", required=True, help="directory for processed txt files")
    return parser.parse_args()


def ellipseRegularized(e):
    e = [float(item) for item in e]
    e[4] = -np.deg2rad(e[4])

    if e[3] > e[2]:
        min_axes = e[2]
        e[2] = e[3]
        e[3] = min_axes
        if e[4] > 0:
            e[4] = e[4] - np.pi / 2
        else:
            e[4] = e[4] + np.pi / 2
    e[2] = e[2] / 2
    e[3] = e[3] / 2
    e[0], e[1] = e[1], e[0]
    return e


def process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue
        gt = os.path.join(input_dir, filename)
        processed_gt = os.path.join(output_dir, filename)
        with open(gt, "r") as f:
            lines = f.readlines()
        num_entries = len(lines) - 1
        with open(processed_gt, "w") as file:
            file.write(str(num_entries) + "\n")
            for line in lines[1:]:
                regularized_line = ellipseRegularized(line.strip().split()[1:])
                line_t = [str(item) + "\t" for item in regularized_line]
                file.write(" ".join(line_t) + "\n")


if __name__ == "__main__":
    args = parse_args()
    process(args.input_dir, args.output_dir)

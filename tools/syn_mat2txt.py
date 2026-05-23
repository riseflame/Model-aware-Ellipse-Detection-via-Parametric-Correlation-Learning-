import argparse
import os

import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(description="Convert synthetic ellipse .mat annotations to txt.")
    parser.add_argument("--mat_dir", required=True, help="directory containing .mat files")
    parser.add_argument("--output_dir", required=True, help="directory for output txt files")
    return parser.parse_args()


def convert(mat_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(mat_dir):
        if not filename.endswith(".mat"):
            continue
        path = os.path.join(mat_dir, filename)
        mat_data = sio.loadmat(path)
        data = mat_data["ellipse_param"]
        write_path = os.path.join(output_dir, "s" + filename[1:-4] + ".jpg.fled.txt")
        with open(write_path, "w") as file:
            file.write(str(data.shape[1]) + "\n")
            for i in range(data.shape[1]):
                column_data = [row[i] for row in data]
                row_str = "\t".join(str(val) for val in column_data)
                file.write(row_str + "\n")


if __name__ == "__main__":
    args = parse_args()
    convert(args.mat_dir, args.output_dir)

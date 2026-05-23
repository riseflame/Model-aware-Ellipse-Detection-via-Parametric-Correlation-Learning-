# Traffic jiaodu
import os
import cv2
import json
import numpy as np
from PIL import Image
import random

def auto_increment_integer_generator():
    i = 1
    while True:
        yield i
        i += 1


if __name__ == '__main__':
    imagelist = os.listdir('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/img100')
    txtlist = os.listdir('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/gt')
    image_id_generator = auto_increment_integer_generator()
    annotation_id_generator = auto_increment_integer_generator()
    images_list = []
    annotation_list = []
        
    for file in txtlist:
        img = "/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/img100/" + file[:-4] +'.jpg'
        pil = Image.open(img)
        width, height = pil.size
        del pil

        if True:
            print(file)
            images_list.append(
                {
                    "license": 1,
                    "file_name": os.path.join(file[:-4] +'.jpg'),
                    "height": height,
                    "width": width,
                    "id": next(image_id_generator)
                }
            )

            bboxs = []
            gt = "/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/gt/"  + file[:-4] + ".txt"
            with open(gt, "r") as f:
                data = f.readlines()
                for i in range(1, int(data[0]) + 1, 1):
                    dic = data[i].split('\t')
                    an = dic[4].split('\n')
                    cx = float(dic[0])
                    cy = float(dic[1])
                    a = float(dic[2])
                    b = float(dic[3])
                    #theta = float(an[0])
                    theta = np.rad2deg(float(an[0]))
                    print(cx, cy, a, b, theta)

                    if a < b:
                        a, b = b, a
                        theta += 90
                    while theta > 90 or theta < -90:
                        if theta > 90:
                            theta -= 180
                        if theta < -90:
                            theta += 180

                    bbox = []
                    bbox.append(cx)
                    bbox.append(cy)
                    bbox.append(a)
                    bbox.append(b)
                    bbox.append(theta)
                    image = images_list[-1]
                    annotation_list.append(
                        {
                            "iscrowd": 0,
                            "image_id": image.get('id'),
                            "bbox": bbox,
                            "category_id": 1,
                            "id": next(annotation_id_generator)
                        }
                    )

    original = dict()
    original['images'] = []
    original['annotations'] = []
    original['images'].extend(images_list)
    original['annotations'].extend(annotation_list)
    print(len(images_list))

    with open('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/test.json', 'w') as f:
        f.write(json.dumps(original, indent=4))
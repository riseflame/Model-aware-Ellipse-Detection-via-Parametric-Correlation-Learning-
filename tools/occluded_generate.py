from PIL import Image, ImageDraw
import random
import cv2
import os
import numpy as np
save_path100 = os.path.join('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/img100/')
save_path80 = os.path.join('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/img80/')
save_path60 = os.path.join('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/img60/')
save_path40 = os.path.join('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/img40/')
txt_save_path = os.path.join('/fastersharefiles/liuzezheng/ElDet-top/tools/sync_crowded/test/gt/')
import cv2
import numpy as np
import random

def generate_ellipses_image(size, num_ellipses,persent,imgname):
    # Create a blank white image
    img100 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    img75 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    img50 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    img25 = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    txtname = txt_save_path+imgname[:-4]+'.txt'
    with open(txtname, 'w') as file:
        file.write(str(num_ellipses) + '\n')
        # Generate random ellipses and draw them on the image
        for _ in range(num_ellipses):
            a = random.randint(10, int(300/np.sqrt(2)))
            ratio = random.randint(10,50)
            b = int(a/ratio*10)
            x = random.randint(0,300)
            y = random.randint(0,300)
            angle = random.randint(0,360)
            rad = np.deg2rad(angle)
            # Draw the ellipse
            cv2.ellipse(img100, (x, y), (a // 2, b // 2), angle, 0, int(360 * 1), (0, 0, 0), 3)
            cv2.ellipse(img75, (x, y), (a // 2, b // 2), angle, 0, int(360 * 0.80), (0, 0, 0), 3)
            cv2.ellipse(img50, (x, y), (a // 2, b // 2), angle, 0, int(360 * 0.60), (0, 0, 0), 3)
            cv2.ellipse(img25, (x, y), (a // 2, b // 2), angle, 0, int(360 * 0.40), (0, 0, 0), 3)
            line = str(x) + '\t' + str(y) + '\t' + str(a//2) + '\t' + str(b//2) + '\t' \
                   + str(rad) + '\n'
            file.write(line)

    return img100,img75,img50,img25

if __name__ == "__main__":
    img_size = (300, 300)
    num_ellipses = 10
    for i in range(120):
        persent = 1
        num_ellipses = i%5 * 4 + 4
        imgname = 'synth_'+str(num_ellipses)+'ellipses_img'+str(i)+'.jpg'

        img100,img80,img60,img40 = generate_ellipses_image(img_size, num_ellipses,persent,imgname)
        cv2.imwrite(save_path100+imgname, img100)
        cv2.imwrite(save_path80+imgname, img80)
        cv2.imwrite(save_path60+imgname, img60)
        cv2.imwrite(save_path40+imgname, img40)

    # Save the generated image


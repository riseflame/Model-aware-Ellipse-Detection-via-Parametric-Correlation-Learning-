import scipy.io as sio
import os
# 加载 .mat 文件
mat_gt = '/fastersharefiles/liuzezheng/Synthetic Images - Occluded Ellipses/gt'
files = os.listdir(mat_gt)
for f in files:
    path = mat_gt + '/' + str(f)
    mat_data = sio.loadmat(path)

# 提取数据，假设保存在变量 data 中
    data = mat_data['ellipse_param']

# 将数据保存为 .txt 文件
    write_path = '/fastersharefiles/liuzezheng/Synthetic Images - Occluded Ellipses/test/'+'s'+str(f)[1:-4]+'.jpg.fled.txt'
    with open(write_path, 'w') as file:
        file.write(str(data.shape[1]) + '\n')
        for i in range(data.shape[1]):
            column_index = i
            column_data = [row[column_index] for row in data]
            row_str = '\t'.join(str(val) for val in column_data)  # 使用制表符分隔每一行的数据
            file.write(row_str + '\n')

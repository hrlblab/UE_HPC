import medmnist
from medmnist import INFO
from PIL import Image
import os
import numpy as np


def save_images_from_npz(npz_path, output_dir, max_per_label=500):
    # 加载 .npz 文件
    data = np.load(npz_path)
    for key in data:
        print(f"Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}")
    images = data['val_images']  # 假设图片存储在 'test_images' 键下
    labels = data['val_labels']  # 假设标签存储在 'test_labels' 键下

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建一个字典来跟踪每个标签的图片数量
    label_count = {}

    # 遍历图片和标签
    for i, (img, label) in enumerate(zip(images, labels)):
        # 获取标签的实际值，如果是单元素数组的话
        label_value = label if np.isscalar(label) else label[0]

        # 初始化标签计数
        if label_value not in label_count:
            label_count[label_value] = 0

        # 检查是否已达到每个标签的限制
        if label_count[label_value] >= max_per_label:
            continue

        # 为每个标签创建一个文件夹
        label_dir = os.path.join(output_dir, str(label_value))
        os.makedirs(label_dir, exist_ok=True)

        # 将图片保存为 .png 格式
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(label_dir, f'{label_value}_{label_count[label_value]}.png'))

        # 更新标签计数
        label_count[label_value] += 1


# 调用函数
npz_path = 'C:\\Users\\67007\Downloads\\Unlearnable-Clusters-main\\Unlearnable-Clusters-main\\organamnist_224.npz'  # .npz 文件路径
output_dir = 'C:\\Users\\67007\\Downloads\\Unlearnable-Clusters-main\\Unlearnable-Clusters-main\\data\\datasets\\classification\\organamnist\\test'  # 输出目录路径
save_images_from_npz(npz_path, output_dir)
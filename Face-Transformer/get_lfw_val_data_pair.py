import os
import random
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_same_pairs=3000, num_diff_pairs=3000):
        self.data_dir = data_dir
        self.class_folders = sorted(os.listdir(data_dir))
        self.image_paths_by_class = {}
        self.num_same_pairs = num_same_pairs
        self.num_diff_pairs = num_diff_pairs
        self.transform = transform
        self._prepare_dataset()

    def _prepare_dataset(self):
        for i, class_folder in enumerate(self.class_folders):
            class_dir = os.path.join(self.data_dir, class_folder)
            image_files = sorted(os.listdir(class_dir))
            if len(image_files) > 1:  # 只有图片数量大于1的类别才加入图像路径列表
                self.image_paths_by_class[i] = [os.path.join(class_dir, image_file) for image_file in image_files]
        self.image_paths_by_class = list(self.image_paths_by_class.values())
    def __getitem__(self, index):
        # class_idx = index % len(self.class_folders)
        # class_images = self.image_paths_by_class[class_idx]

        if index < self.num_same_pairs:
            # 生成相同类别的样本对
            class_idx = random.choice(range(len(self.image_paths_by_class)))
            class_images = self.image_paths_by_class[class_idx]
            image1_path = random.choice(class_images)
            image2_path = random.choice(class_images)
            while image1_path == image2_path: # 确保不同样本
                image2_path = random.choice(class_images)
            label = 1  # 相同类别
        else:
            class_idx1 = random.choice(range(len(self.class_folders)))
            # 生成不同类别的样本对
            class_idx2 = random.choice(range(len(self.class_folders)))
            while class_idx1 == class_idx2: # 确保不同类别
                class_idx2 = random.choice(range(len(self.class_folders)))
            class_folder1 = self.class_folders[class_idx1]
            class_folder2 = self.class_folders[class_idx2]
            
            # 从两个class_folder1,class_folder2文件夹中分别随机选择一个样本
            class_list1 = os.listdir(os.path.join(self.data_dir, class_folder1))
            class_list2 = os.listdir(os.path.join(self.data_dir, class_folder2))
            image1_path = os.path.join(self.data_dir, class_folder1, random.choice(class_list1))
            image2_path = os.path.join(self.data_dir, class_folder2, random.choice(class_list2))
           
            label = 0  # 不同类别

        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")
        print(image1_path)
        print(image2_path)
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            
        return image1, image2, label

    def __len__(self):
        return self.num_same_pairs + self.num_diff_pairs


if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset('/data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/data/lfw-deepfunneled',transform=transform)
    import torch
    a=dataset[2]
    a[0]
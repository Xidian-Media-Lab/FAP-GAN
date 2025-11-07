import torch
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.data import Dataset
import os
from os import listdir
from PIL import Image
import glob
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
import json

from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.LOAD_TRUNCATED_IMAGES = True




class RGB_ycbcr(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_ycbcr, self).__init__()
        self.eps = eps

    def rgb_to_ycbcr(self, img):
        if len(img.shape) != 4:
            raise ValueError('Input image must have four dimension,not %d' %len(img.shape))
        y = 0. + 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
        cb = 0.5 - 0.156 * img[:, 0] - 0.331 *img[:, 1] + 0.5 * img[:, 2]
        cr = 0.5 + 0.5 * img[:, 0] - 0.419 *img[:, 1] - 0.081 * img[:, 2]
        return torch.cat((y.unsqueeze(1),cb.unsqueeze(1),cr.unsqueeze(1)),1)

        return hsv

    def yccbr_to_rgb(self, img):
        if len(img.shape) != 4:
            raise ValueError('Input image must have four dimension,not %d' %len(img.shape))
        r = img[:, 0] + 1.4 * (img[:, 2] - 0.5)
        g = img[:, 0] - 0.343 * (img[:, 1]- 0.5) - 0.711 * (img[:, 2] - 0.5)
        b = img[:, 0] + 1.765 * (img[:, 1] - 0.5)
        return torch.cat((r.unsqueeze(1),g.unsqueeze(1),b.unsqueeze(1)),1)

def rgb_to_yuv420(rgb):
    rgb_array = np.array(rgb)
    H, W, _ = rgb_array.shape
    yuv_array = np.empty((H, W, 3), dtype = np.uint8)

    yuv_array[:, :, 0] = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
    yuv_array[:, :, 1] = -0.14713 * rgb_array[:, :, 0] - 0.288862 * rgb_array[:, :, 1] + 0.436 * rgb_array[:, :, 2]
    yuv_array[:, :, 2] = 0.615 * rgb_array[:, :, 0] - 0.51498 * rgb_array[:, :, 1] - 0.10001 * rgb_array[:, :, 2]

    return Image.fromarray(yuv_array)




class TrainImageFolder(datasets.ImageFolder):
    # def __init__(self, imagedir, gtdir, transform=True):

    def __init__(self, LR22train_dir, LR32train_dir, LR42train_dir, LR52train_dir, LRtrain_dir,  transform=None):   #with crop
        # super(TrainImageFolder, self).__init__(nirdir, size_crop, stride_crop)
        self.img_index = 233
        self.qp22_path = LR22train_dir
        self.qp32_path = LR32train_dir
        self.qp42_path = LR42train_dir
        self.qp52_path = LR52train_dir
        self.gt_path = LRtrain_dir

        self.image22_list = self.get_image_list(self.qp22_path, self.img_index)
        self.image32_list = self.get_image_list(self.qp32_path, self.img_index)
        self.image42_list = self.get_image_list(self.qp42_path, self.img_index)
        self.image52_list = self.get_image_list(self.qp52_path, self.img_index)
        self.imagegt_list = self.get_image_list(self.gt_path, self.img_index)
        self.ref_img_list = self.get_ref(self.img_index)

        self.transform = transform

    def pathfilter(self, path1,path2, path3, path4,path5):
        # 定义四个路径列表
        file_dict = {}
        all_paths = [path1, path2, path3, path4, path5]
        for path_list in all_paths:
            for path in path_list:
                filename = os.path.basename(path)
                if filename in file_dict:
                    file_dict[filename].append(path)
                else:
                    file_dict[filename] = [path]

        # 找出在所有路径列表中都存在的文件名
        common_files = [filename for filename, paths in file_dict.items() if len(paths) == len(all_paths)]

        # 创建四个新的列表，包含公共文件名的路径
        new_path1 = [path for path in path1 if os.path.basename(path) in common_files]
        new_path2 = [path for path in path2 if os.path.basename(path) in common_files]
        new_path3 = [path for path in path3 if os.path.basename(path) in common_files]
        new_path4 = [path for path in path4 if os.path.basename(path) in common_files]
        new_path5 = [path for path in path5 if os.path.basename(path) in common_files]

        output_data = {
            "new_path1": new_path1,
            "new_path2": new_path2,
            "new_path3": new_path3,
            "new_path4": new_path4,
            "new_path5": new_path5,

        }

        with open('filtered_paths.json', 'w') as f:
            json.dump(output_data, f, indent=4)

        print("路径列表已保存到文件 filtered_paths.json")




        return new_path1,new_path2, new_path3, new_path4,new_path5

    def safe_load_image(self, path, size):
        try:
            with Image.open(path) as img:
                return img.convert('RGB').resize(size).copy()
        except OSError as e:
            print(f"[warning] Failed to open image {path}: {e}")
            return None

    def get_image_list(self, root_folder, img_index):
        image_list = []
        # root_folder = self.gt_path
        for i in range(0,img_index):
            folder_path = os.path.join(root_folder, f"video_{i}")
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    file = os.path.join(folder_path, f)

                    image_list.append(file)
            print(i)

        return image_list

    def get_ref(self, img_index):
        ref_list = []
        root_folder = self.gt_path
        for i in range(0, img_index):
            folder_path = os.path.join(root_folder, f"video_{i}")
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    file = os.path.join(folder_path, "frame"+ "_0001.png")
                    # file = os.path.join(folder_path, "video_"+ str(i) + "_0001.png")
                    ref_list.append(file)
        return ref_list

    def __len__(self):
        return len(self.image22_list)


    def __getitem__(self, index):
        H = 512
        W = 512

        New_size = (H,W)

        qp22_path = self.image22_list[index]
        qp32_path = self.image32_list[index]
        qp42_path = self.image42_list[index]
        qp52_path = self.image52_list[index]
        gt_path = self.imagegt_list[index]
        ref_path = self.ref_img_list[index]


        img_paths = [qp22_path, qp32_path, qp42_path, qp52_path, gt_path, ref_path]
        # print(img_paths)

        images = [self.safe_load_image(p, New_size) for p in  img_paths]

        if any([img is None for img in images]):
            return None

        qp22_img, qp32_img, qp42_img, qp52_img, gt_img, ref_img = images

        qp22_img = qp22_img.resize(New_size)
        qp32_img = qp32_img.resize(New_size)
        qp42_img = qp42_img.resize(New_size)
        qp52_img = qp52_img.resize(New_size)
        gt_img = gt_img.resize(New_size)
        ref_img = ref_img.resize(New_size)

        # gt_img = rgb_to_yuv420(gt_img)
        # qp22_img = rgb_to_yuv420(qp22_img)
        # qp32_img = rgb_to_yuv420(qp32_img)
        # qp42_img = rgb_to_yuv420(qp42_img)
        # qp52_img = rgb_to_yuv420(qp52_img)
        # ref_img = rgb_to_yuv420(ref_img)

        qp22_img = self.transform(qp22_img)
        qp32_img = self.transform(qp32_img)
        qp42_img = self.transform(qp42_img)
        qp52_img = self.transform(qp52_img)
        gt_img = self.transform(gt_img)
        ref_img = self.transform(ref_img)
        

        return qp22_img, qp32_img, qp42_img, qp52_img, gt_img, ref_img


class DatasetRepeater(Dataset):
    def __init__(self, dataset, num_repeaters=100):
        self.dataset = dataset
        self.num_repeaters = num_repeaters

    def __len__(self):
        return self.num_repeaters * len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def dataloader(batch_size, num_workers):
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=0.485,std=0.225)]
    # )

    transform = transforms.ToTensor()

    LR22train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRdecoded/22/"
    LR32train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRdecoded/32/"
    LR42train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRdecoded/42/"
    LR52train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRdecoded/52/"
    LRtrain_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRoriginal/"

    # LR22train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/3090ti/liulu/LRdecoded/22/"
    # LR32train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/3090ti/liulu/LRdecoded/32/"
    # LR42train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/3090ti/liulu/LRdecoded/42/"
    # LR52train_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/3090ti/liulu/LRdecoded/52/"
    # LRtrain_dir = "/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/3090ti/liulu/LRoriginal/"
    

    train_dataset = TrainImageFolder(LR22train_dir, LR32train_dir, LR42train_dir, LR52train_dir, LRtrain_dir, transform=transform)
    train_dataset = DatasetRepeater(train_dataset, num_repeaters=12)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True, num_workers = 0, collate_fn= collate_fn_skip_none)
    return train_loader


if __name__ == "__main__":
    data = dataloader(8, 8)
    print('finish')
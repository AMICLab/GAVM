import os
import random
from pathlib import Path
import cv2
import Augmentor
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import utils
from torch.utils import data


def load_npy_data(file_path):
    if not file_path.endswith('.npy'):
        raise ValueError("File extension is not .npy, please provide correct file path.")

    data = np.load(file_path)
    data = data.squeeze(0)
    return data


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff', 'nii', 'npy'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample

        if condition == 1:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
        elif condition == 0:
            self.paths = self.load_flist(folder)
        elif condition == 2:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])

        self.image_size = image_size
        self.convert_image_to = convert_image_to

    def __len__(self):
        if self.condition:
            return len(self.input)
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.condition == 1:
            img0 = load_npy_data(self.gt[index])
            img1 = load_npy_data(self.input[index])

            img0 = convert_image_to_fn(self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(self.convert_image_to, img1) if self.convert_image_to else img1

            img0, img1 = self.pad_img([img0, img1], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)
            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            normalization_factor = 15000.0
            img0 = img0 / normalization_factor
            img1 = img1 / normalization_factor
            return [self.to_tensor(img0), self.to_tensor(img1)]

        elif self.condition == 0:
            path = self.paths[index]
            img = Image.open(path)
            img = convert_image_to_fn(self.convert_image_to, img) if self.convert_image_to else img
            img = self.pad_img([img], self.image_size)[0]
            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]
            img = self.cv2equalizeHist(img) if self.equalizeHist else img
            images = [[img]]
            p = Augmentor.DataPipeline(images)

            if self.augment_flip:
                p.flip_left_right(1)

            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)

            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            return self.to_tensor(img)

        elif self.condition == 2:
            img0 = load_npy_data(self.gt[index])
            img1 = load_npy_data(self.input[index])
            img2 = load_npy_data(self.input_condition[index])

            img0 = convert_image_to_fn(self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(self.convert_image_to, img1) if self.convert_image_to else img1
            img2 = convert_image_to_fn(self.convert_image_to, img2) if self.convert_image_to else img2

            img0, img1, img2 = self.pad_img([img0, img1, img2], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch([img0, img1, img2], self.image_size)

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1, img2]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)

            g = p.generator(batch_size=1)
            augmented_images = next(g)

            img0 = augmented_images[0][0]
            img1 = augmented_images[0][1]
            img2 = augmented_images[0][2]

            normalization_factor = 15000.0
            img0 = img0 / normalization_factor
            img1 = img1 / normalization_factor
            img2 = img2 / normalization_factor
            return [self.to_tensor(img0), self.to_tensor(img1), self.to_tensor(img2)]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [str(p) for ext in self.exts for p in Path(flist).glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print(f"Error loading file list from {flist}: {e}")
                    return [flist]

        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img_eq = cv2.merge((b, g, r))
        return img_eq

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir + "_" + os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        h, w = image_list[0].shape[:2]

        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)

        patched_images = []
        for img in image_list:
            if len(img.shape) == 2:
                patch = img[rr:rr + patch_size, cc:cc + patch_size]
            else:
                patch = img[rr:rr + patch_size, cc:cc + patch_size, :]
            patched_images.append(patch)

        return patched_images

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            h, w = img.shape[:2]

            bottom, right = 0, 0

            if h < patch_size:
                bottom = patch_size - h
                h = patch_size

            if w < patch_size:
                right = patch_size - w
                w = patch_size

            bottom = bottom + (h // block_size) * block_size + \
                     (block_size if h % block_size != 0 else 0) - h

            right = right + (w // block_size) * block_size + \
                    (block_size if w % block_size != 0 else 0) - w

            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            i += 1

        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size - h
            h = patch_size
        if w < patch_size:
            right = patch_size - w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
                 (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]


def save_grayscale_images(tensor, filename):
    tensor = tensor.detach().cpu().numpy()
    imgs = np.squeeze(tensor, axis=1)

    batch_size, height, width = imgs.shape
    canvas_height = int(np.ceil(np.sqrt(batch_size)) * height)
    canvas_width = int(np.ceil(batch_size / (canvas_height / height)) * width)
    canvas = Image.new('L', (canvas_width, canvas_height))

    for idx, img in enumerate(imgs):
        row = idx // (canvas_width // width)
        col = idx % (canvas_width // width)
        img_pil = Image.fromarray(np.uint8(img * 255), 'L')
        canvas.paste(img_pil, (col * width, row * height))

    canvas.save(filename)


def show_images(images, title="Images"):
    grid = utils.make_grid(images, nrow=2, padding=2, normalize=True).permute(1, 2, 0).numpy()
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    folder = ['/home/siat/img/test.txt', '/home/siat/label/test.txt', '/home/siat/mri/test.txt']
    dataset = Dataset(folder, 256, augment_flip=False,
                      convert_image_to=None, condition=2, equalizeHist=False,
                      crop_patch=True)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        images, labels, mri = sample_batched
        print(images.max())
        show_images(images, "Input Images")
        show_images(labels, "Label Images")
        show_images(mri, "mri Images")
        break
import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    if not mask_file:
        raise FileNotFoundError(f"No files found for {idx} with suffix {mask_suffix} in {mask_dir}")
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str,images_trans_dir:str, mask_trans_dir:str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.images_trans_dir = Path(images_trans_dir)
        self.mask_trans_dir = Path(mask_trans_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # 检查mask包含的值生成对应标签(比较耗时,暂时去掉,但要保证数据的正确性)
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # self.mask_values = [[0, 0, 0], [255, 255, 255]]
        # self.mask_values = [0,255]

        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        mask_trans_file = list(self.mask_trans_dir.glob(name + self.mask_suffix + '.*'))
        img_trans_file = list(self.images_trans_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        # 验证file not find是否真的是文件不存在
        # mask = load_image(mask_file[0])
        # img = load_image(img_file[0])
        # mask_trans = load_image(mask_trans_file[0])
        # img_trans = load_image(img_trans_file[0])
        try:
            mask = load_image(mask_file[0])
        except Exception as e:
             logging.info(f"[Error] Failed to load image {mask_file[0]}. Exception: {e}")
        try:
            img = load_image(img_file[0])
        except Exception as e:
             logging.info(f"[Error] Failed to load image {img_file[0]}. Exception: {e}")
        try:
            mask_trans = load_image(mask_trans_file[0])
        except Exception as e:
             logging.info(f"[Error] Failed to load image {mask_trans_file[0]}. Exception: {e}")
        try:
            img_trans = load_image(img_trans_file[0])
        except Exception as e:
             logging.info(f"[Error] Failed to load image {img_trans_file[0]}. Exception: {e}")



        # 检查标签是否全为0
        mask_array = np.array(mask)
        mask_trans_array = np.array(mask_trans)
        # if np.all(mask_array == 0) or np.all(mask_trans_array == 0):
        #     print(f"The image '{name}' is completely black (all pixels are [0, 0, 0]).")
        assert not np.all(mask_array == 0) and not np.all(mask_trans_array == 0),f'The image {name} is completely black (all pixels are [0, 0, 0]).'


        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        img_trans = self.preprocess(self.mask_values, img_trans, self.scale, is_mask=False)
        mask_trans = self.preprocess(self.mask_values, mask_trans, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_trans': torch.as_tensor(img_trans.copy()).float().contiguous(),
            'mask_trans': torch.as_tensor(mask_trans.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir,images_trans_dir, mask_trans_dir, scale=1):
        super().__init__(images_dir, mask_dir,images_trans_dir, mask_trans_dir, scale, mask_suffix='')

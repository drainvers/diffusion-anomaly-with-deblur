import math
import random
import os
from pathlib import Path
from PIL import Image
import blobfile as bf
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .train_util import visualize
from visdom import Visdom
viz = Visdom(port=8850)
from scipy import ndimage


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    classes = None

    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.

        class_names =[path.split("/")[-2] for path in all_files] #9 or 3
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names), reverse=True))}
        if len(sorted_classes) == 1 and sorted_classes.get('diseased') == 0:
                sorted_classes['diseased'] = 1
        print('sorted_classes', sorted_classes)
        classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    print('lenloader', len(loader))
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = os.path.splitext(os.path.basename(full_path))[1]
        if ext.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        print('len',  len(self.local_images))
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        # name=str(path).split("/")[-1].split(".")[0]
        name, ext = os.path.splitext(os.path.basename(path))
        print('name', name)

        # Readds ability to read known image formats, taken from upstream guided diffusion repo
        if ext == '.npy':
            numpy_img = np.load(path)
        else:
            numpy_img = np.asarray(Image.open(path).convert('L')) # Use Pillow for TIF support
            # numpy_img = cv2.resize(numpy_img, (256, 256), interpolation=cv2.INTER_AREA)
            numpy_img = np.expand_dims(numpy_img, axis=2) # Changes image shape to (W, H, 1)
        arr = visualize(numpy_img).astype(np.float32)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["name"]=name

        return np.transpose(arr, [2, 0, 1]), out_dict # HWC -> CHW


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 3* image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
   # crop_y=64; crop_x=64
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def zeropatch(pil_image, image_size):
    im=np.array(th.zeros(image_size, image_size,3))
    arr = np.array(pil_image)
    crop_x = (-arr.shape[0] + image_size)
    crop_y = abs(arr.shape[1] - image_size) // 2
  #  print('crop', crop_y, crop_x) #crop_y=64; crop_x=64
    im[0:arr.shape[0] , crop_y : crop_y +arr.shape[1],:]=arr

    return im#arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

"""Some parts based on https://github.com/yang-song/score_sde_pytorch"""

from torch.utils.data import DataLoader, Dataset
import numpy as np
from mpi4py import MPI
import blobfile as bf
from torchvision import transforms, datasets
import torch
import cv2

class UniformDequantize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return (torch.rand(sample.shape) + sample*255.)/256.


def get_dataset(config, uniform_dequantization=False, train_batch_size=None,
                eval_batch_size=None, num_workers=0):
    """
    Get Pytorch dataloaders for one of the following datasets:
    MNIST, CIFAR-10, LSUN-Church, FFHQ, AFHQ
    MNIST and CIFAR-10 are loaded through torchvision, others have to be
    downloaded separately to the data/ folder from the following sources:
    https://github.com/NVlabs/ffhq-dataset
    https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq
    https://github.com/fyu/lsun
    """

    transform = [transforms.Resize(config.data.image_size),
                 transforms.CenterCrop(config.data.image_size)]
    if config.data.random_flip:
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    if uniform_dequantization:
        transform.append(UniformDequantize())
    transform = transforms.Compose(transform)

    if not train_batch_size:
        train_batch_size = config.training.batch_size
    if not eval_batch_size:
        eval_batch_size = config.eval.batch_size
    
    if config.data.dataset == 'MNIST':
        training_data = datasets.MNIST(
            root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=transform)
    elif config.data.dataset == 'SimuData':
        trainloader = load_data(data_dir="data/SimuData/train",
                                batch_size=train_batch_size, image_size=config.data.image_size,
                                random_flip=config.data.random_flip)
        testloader = load_data(data_dir="data/SimuData/test",
                               batch_size=eval_batch_size, image_size=config.data.image_size,
                               random_flip=False)
        return trainloader, testloader
    elif config.data.dataset == 'BioSR':
        trainloader = load_data(data_dir="data/BioSR/train",
                                batch_size=train_batch_size, image_size=config.data.image_size,
                                random_flip=config.data.random_flip)
        testloader = load_data(data_dir="data/BioSR/test",
                               batch_size=eval_batch_size, image_size=config.data.image_size,
                               random_flip=False, random_enhance=False)
        WFloader = load_data(data_dir="data/BioSR/WF",
                                  batch_size=eval_batch_size, image_size=config.data.image_size,
                                  random_flip=False, random_enhance=False)
        return trainloader, testloader, WFloader
    elif config.data.dataset == 'STED':
        trainloader = load_data(data_dir="data/STED/train",
                                batch_size=train_batch_size, image_size=config.data.image_size,
                                random_flip=config.data.random_flip)
        testloader = load_data(data_dir="data/STED/test",
                               batch_size=eval_batch_size, image_size=config.data.image_size,
                               random_flip=False, random_enhance=False)
        WFloader = load_data(data_dir="data/STED/WF",
                                  batch_size=eval_batch_size, image_size=config.data.image_size,
                                  random_flip=False, random_enhance=False)
        return trainloader, testloader, WFloader
    else:
        raise ValueError

    # If we didn't use the load_data function that already created data loaders:
    trainloader = DataLoader(training_data, batch_size=train_batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=eval_batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    return trainloader, testloader


""" The following mostly pasted from the Improved Denoising Diffusion models github page:
		https://github.com/openai/improved-diffusion """


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
    random_flip=True, random_enhance=True, drop_last=True, return_paths=False,
    shard=None, num_shards=None
):
    """
    NOTE: Change to original function, returns the Pytorch dataloader, not a generator

    For a dataset, create a dataloader over (images, kwargs) pairs.
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
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    if shard is None:
        shard = MPI.COMM_WORLD.Get_rank()
    if num_shards is None:
        num_shards = MPI.COMM_WORLD.Get_size()

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=shard,
        num_shards=num_shards,
        random_flip=random_flip,
        random_enhance=random_enhance
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=drop_last
        )
    if return_paths:
        return loader, dataset.local_images
    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif", "tiff"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1,
                 random_flip=True, random_enhance=True, brightness_range=(0.5, 2.0)):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_enhance = random_enhance
        self.brightness_range = brightness_range

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        image_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)        
        
        # Convert the 16-bit image to float32 for preprocessing.
        gray_image = image_array.astype(np.float32)  
        
        # Resize the image with OpenCV.
        while min(*gray_image.shape[:2]) >= 2 * self.resolution:
            gray_image = cv2.resize(
                gray_image,
                (gray_image.shape[1] // 2, gray_image.shape[0] // 2),
                interpolation=cv2.INTER_AREA
            )

        scale = self.resolution / min(*gray_image.shape[:2])
        gray_image = cv2.resize(
            gray_image,
            (round(gray_image.shape[1] * scale), round(gray_image.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC
        )

        if self.random_flip:
            k = np.random.randint(0, 4)
            if k > 0:
                gray_image = np.rot90(gray_image, k)
            if np.random.rand() > 0.5:
                gray_image = np.fliplr(gray_image)
            if np.random.rand() > 0.5:
                gray_image = np.flipud(gray_image)
                
        # Apply random brightness scaling.
        if self.random_enhance is not None and self.random_enhance is True:
            factor = np.random.uniform(*self.brightness_range)
            gray_image = np.clip(gray_image * factor, 0, 65535)
            
            
        image_array = np.array(gray_image)
        crop_y = (image_array.shape[0] - self.resolution) // 2
        crop_x = (image_array.shape[1] - self.resolution) // 2
        image_array = image_array[crop_y: crop_y + self.resolution,
                  crop_x: crop_x + self.resolution]

        # Normalize pixel intensities to the [0, 1] range.
        image_array = image_array / 65535.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        return np.transpose(image_array[np.newaxis, :, :], [0, 1, 2]), out_dict
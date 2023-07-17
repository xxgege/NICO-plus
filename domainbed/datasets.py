# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageFile
from os.path import join
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _dataset_info(txt_file):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(' '.join(row[:-1]))
        labels.append(int(row[-1]))

    return file_names, labels


class StandardDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

def get_train_transformer(): # hard-coded
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transformer(): # hard-coded
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloader(txtdir, dataset, domain, phase, batch_size, num_workers=8):
    assert phase in ["train", "val", "test"]
    names, labels = _dataset_info(join(txtdir, dataset, "%s_%s.txt"%(domain, phase)))

    if phase == "train":
        img_tr = get_train_transformer()
    else:
        img_tr = get_val_transformer()
    curDataset = StandardDataset(names, labels, img_tr)
    if phase == "train":
        loader = InfiniteDataLoader(dataset=curDataset, weights=None, batch_size=batch_size, num_workers=num_workers)
    else:
        loader = FastDataLoader(dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
    return loader

def get_mix_dataloader(txtdir, dataset, domains, phase, batch_size, num_workers=8):
    assert phase == "train"
    img_tr = get_train_transformer()
    concat_list = []
    for domain in domains:
        names, labels = _dataset_info(join(txtdir, dataset, "%s_%s.txt"%(domain, phase)))
        curDataset = StandardDataset(names, labels, img_tr)
        concat_list.append(curDataset)
    finalDataset = data.ConcatDataset(concat_list)
    loader = InfiniteDataLoader(dataset=finalDataset, weights=None, batch_size=batch_size, num_workers=num_workers)
    return loader
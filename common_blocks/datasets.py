import cv2
import skimage.io
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.df = df
        self.labels = df[config['target_col']]
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = '{}/{}.{}'.format(self.config['images_path'], file_name, self.config['images_type'])
        image = skimage.io.MultiImage(file_path)
        image = cv2.resize(image[-1], (self.config['target_width'], self.config['target_width']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]

        return image, float(label)


class TestDataset(Dataset):
    # todo test and rewrite for kaggle inference

    def __init__(self, df, dir_name, config, transform=None):
        self.df = df
        self.dir_name = dir_name
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = '{}/{}.{}'.format(self.config['images_path'], file_name, self.config['images_type'])
        image = skimage.io.MultiImage(file_path)
        image = cv2.resize(image[-1], (self.config['target_width'], self.config['target_width']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image


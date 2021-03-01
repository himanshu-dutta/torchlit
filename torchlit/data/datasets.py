import os
import PIL

import torch
from torch.utils.data import Dataset


class ImageDatasetFromDF(Dataset):
    r"""Creates image dataset from a `pandas.DataFrame` instance.

    Parameters
    ----------
    - `root_directory`: str
        directory for image files
    - `dataframe`: pandas.DataFrame
        dataframe object with image filename and labels.
    - `image_file_col`: str
        column name of image filename.
    - `label_col`: str
        column name of label column
    - `mapping`: dict
        labels to class idx mapping
    """

    def __init__(
        self,
        root_directory,
        dataframe,
        image_file_col,
        label_col,
        mapping,
        transforms=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.dir = root_directory
        self.tfrm = transforms
        self.labels = label_col
        self.ifl = image_file_col

        self.class_to_label = mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = os.path.join(str(self.dir), str(self.df.loc[index, self.ifl]))
        labels = torch.tensor(
            self.class_to_label[self.df.loc[index, self.labels]], dtype=torch.float32
        )
        image = PIL.Image.open(filename)
        if self.tfrm:
            image = self.tfrm(image)

        return image, labels


class ImageDatabunch(Dataset):
    r"""Creates an image dataset from `img_dir` parameter. Useful for making prediction on a image folder.
    Parameters
    ----------
    - `root_directory`: str
        directory for image files
    """

    def __init__(self, img_dir, transforms=None):
        self.dir = img_dir
        self.image_files = [
            f
            for f in os.listdir(img_dir)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
        ]
        self.tfrm = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = os.path.join(str(self.dir), str(self.image_files[index]))
        image = PIL.Image.open(filename)
        if self.tfrm:
            image = self.tfrm(image)

        return image, str(self.image_files[index])

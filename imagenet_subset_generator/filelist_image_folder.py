
import numpy as np
import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize

class FilelistImageFolder(Dataset):
    def __init__(self, data_folder_uri, filelist_uri):
        super().__init__()
        with open(filelist_uri) as f:
            filelist = f.readlines()
        self.filelist = []
        for line in filelist:
            split = line.split(" ")
            if len(split) == 3:
                self.filelist.append((split[0], split[1], int(split[2])))
        self.data_folder_uri = data_folder_uri

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        classname, imgname, y = self.filelist[index]
        with open(f"{self.data_folder_uri}/{classname}/{imgname}", 'rb') as f:
            img = Image.open(f)
            data = img.convert('RGB')
        return to_tensor(resize(data, (224, 224))), y


import io
from PIL import Image
from torch.utils.data import Dataset
import h5py
from torchvision.transforms.functional import to_tensor

class H5ImageFolder(Dataset):
    def __init__(self, h5_file_uri, filelist_uri):
        super().__init__()
        with open(filelist_uri) as f:
            filelist = f.readlines()
        self.filelist = []
        for line in filelist:
            split = line.split(" ")
            if len(split) == 3:
                self.filelist.append((split[0], split[1], int(split[2])))
        self.h5_file = h5py.File(h5_file_uri, "r")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        classname, imgname, y = self.filelist[index]
        img = self.h5_file[classname][imgname][:]
        img = Image.open(io.BytesIO(img)).convert("RGB")
        return to_tensor(img), y


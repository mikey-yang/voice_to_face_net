import copy
from .voxceleb_dataset import VoxCeleb
from base import BaseDataLoader
from torch.utils.data import DataLoader


class SpeakerID(BaseDataLoader):
    """
    VoxCeleb Dataloader to run a SpeakerID network 

        data_dir --> list of all files (iden_split.txt)
    """
    def __init__(self, data_dir, specgrams_dir, root_path, file_extension, batch_size, shuffle=True, validation_split=0.0, num_workers=1, test=False):
        self.data_dir = data_dir 
        self.root_path = root_path
        self.file_extension = file_extension
        self.specgrams_dir = specgrams_dir
        self.mode = "test" if test else "train"
        self.dataset = VoxCeleb(self.data_dir, self.specgrams_dir, self.root_path, self.file_extension, self.mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=None)

    """
        This function will return a DataLoader for validation
        WARNING: it will override the validation split indicated in the config.json 
    """
    def split_validation(self):
        val_dataset = VoxCeleb(self.data_dir, self.specgrams_dir, self.root_path, self.file_extension, "test")
        init_kwargs = copy.deepcopy(self.init_kwargs)
        init_kwargs["dataset"] = val_dataset

        return DataLoader(**init_kwargs)




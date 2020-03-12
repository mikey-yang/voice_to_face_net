import os
import numpy as np
from PIL import Image
from random import randint
from torch.utils.data import Dataset
from collections import defaultdict


class VoxCelebVGGFace(Dataset):
    """
        This dataloader loads VoxCeleb and VGGFace simultaneously 

    """
    def __init__(self, dataset_file, dataset_types, model_mode, segment_length=400):
        types = {"train": 1, "val": 2, "test": 3}
        mode = {"voice": 1, "face": 2, "dual": 3}
        self.label_types = [types[i] for i in dataset_types]
        self.segment_length = segment_length
        self.dataset, self.labels = self.read_dataset(dataset_file)
        self.model_mode = mode[model_mode]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        split, utt, face, y = self.dataset[index]
        X_face = None
        X_voice = None

        if (self.model_mode == 1 or self.model_mode == 3):
            # load the audio of length = segment_length
            x = np.load(utt)
            recording_length = x.shape[1]
            new_x = np.zeros((257, self.segment_length))
            if recording_length > self.segment_length:
                start = randint(0, recording_length-self.segment_length)
                end = start+self.segment_length
            else:
                start = 0
                end = recording_length
            new_x[:, :end-start] = x[:, start:end]
            X_voice = new_x
            
        if (self.model_mode == 2 or self.model_mode == 3):
            # load the face 
            face_pixels = np.array(Image.open(face), np.float).flatten()
            X_face = face_pixels

        # get the label for the person's ID
        Y = self.labels.index(y)
        
        # return voice and label for mode == "voice"
        if (self.model_mode == 1):
            return X_voice, Y
        
        # return face and label for mode == "face"
        if (self.model_mode == 2):
            return X_face, Y
        
        # return voice, face and label for mode == "dual"
        if (self.model_mode == 3):
            return X_voice, X_face, Y

    def read_dataset(self, dataset_file):
        dataset = []
        with open(dataset_file, "r") as f:
            for l in f.readlines():
                file_meta = l.strip().split(",")
                file_meta[0] = int(file_meta[0])
                if file_meta[0] not in self.label_types:
                    continue
                dataset.append(file_meta)
        # generate the labels so it is consistent with the speaker ID 
        labels = ['id'+str(10000+i) for i in range(1, 1252)]
        return dataset, labels


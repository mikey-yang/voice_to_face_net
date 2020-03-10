import os
import numpy as np
from random import randint
from torch.utils.data import Dataset
from collections import defaultdict

class VoxCeleb(Dataset):
    def __init__(self, data_dir, specgrams_dir, root_path, file_extension, dataset_type, segment_length=400):
        types = {"train": 1, "val": 2, "test": 3}
        label_type = types[dataset_type]
        self.segment_length = segment_length
        self.dataset, self.labels = self.read_dataset(data_dir, specgrams_dir, root_path, file_extension, label_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        # try:
        x = np.load(x)
        # except:
        #     x = np.zeros((257,self.segment_length))
        y = self.labels.index(y)
        recording_length = x.shape[1]
        new_x = np.zeros((257, self.segment_length))
        if recording_length > self.segment_length:
            start = randint(0, recording_length-self.segment_length)
            end = start+self.segment_length
        else:
            start = 0
            end = recording_length
        new_x[:, :end-start] = x[:, start:end]
        return new_x, y

    def read_dataset(self, data_dir, specgrams_dir, root_path, file_extension, label_type):
        all_specgrams = defaultdict(lambda:len(all_specgrams))
        all_files = []
        labels = set()
        if label_type == 1:
            label_type = [1,2]
        else:
            label_type = [3]
        with open(specgrams_dir, "r") as f:
            for line in f.readlines():
                all_specgrams[line.strip()]
        with open(data_dir, "r") as f:
            for line in f.readlines():
                label, filepath = line.strip().split(" ")
                label = int(label)
                if label not in label_type:
                    continue
                filepath, ext = os.path.splitext(filepath)
                filepath += "." + file_extension
                speaker_id = filepath.split("/")[0]
                labels.add(speaker_id)
                filepath = os.path.join(root_path, filepath)
                # too slow to call on workhorse...
                # if not os.path.isfile(filepath):
                #     continue
                if filepath not in all_specgrams:
                    continue
                all_files.append((filepath, speaker_id))

        labels_to_return = sorted(list(labels))
        with open("/home/mahmoudi/workhorse3/voice_to_face_net/template_labels.txt", "w") as f:
            for x in labels_to_return:
                f.write(x+"\n")
        print("exported the labels, exiting now...")
        return all_files, labels_to_return 

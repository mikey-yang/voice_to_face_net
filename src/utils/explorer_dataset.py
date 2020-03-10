import os 
import sys 

# ORIGINAL DATASPLIT 
ALL_FILES = "/share/workhorse3/mahmoudi/voxceleb/dataset/iden_split.txt"
# MY DATASPLIT 
MY_FILES = "/share/workhorse3/mahmoudi/voxceleb/dataset/all_specgrams.txt" 

def get_dataset_stats():
    mapping = {1: {}, 2: {}, 3:{}}

    with open(ALL_FILES, "r") as f:
        for l in f.readlines():
            label, fpath = l.strip().split(" ")
            label = int(label)
            userid = fpath.split("/")[0]
            if userid in mapping[label]:
                mapping[label][userid].append(fpath)
            else:
                mapping[label][userid] = [fpath]

    print("POIs in eachs split: {},{},{}".format(len(mapping[1]), len(mapping[2]), len(mapping[3])))
    train_n_files = sum([len(v) for k,v in mapping[1].items()])
    val_n_files = sum([len(v) for k,v in mapping[2].items()])
    test_n_files = sum([len(v) for k,v in mapping[3].items()])

    print("Number of files per train split: {}".format(train_n_files))
    print("Number of files per val split: {}".format(val_n_files))
    print("Number of files per test split: {}".format(test_n_files))

    return mapping

def get_specgram_stats():
    dataset_mapping = get_dataset_stats()
    dataset_flipped = {}

    for split,user in dataset_mapping.items():
        for user, files in user.items():
            for f in files:
                dataset_flipped[f] = {"label": split, "user": user}

    root = "/share/workhorse3/mahmoudi/voxceleb/wav_specgram/"

    users = {1: set(), 2: set(), 3: set()}
    n_files = 0
    with open(MY_FILES, "r") as f:
        for l in f.readlines():
            n_files += 1
            fpath = l.strip().replace(root, "")
            fpath = os.path.splitext(fpath)[0] + ".wav"
            userid = fpath.split("/")[0]
            metadata = dataset_flipped[fpath]
            users[metadata["label"]].add(metadata["user"])

    print("Number of users in train split: {}".format(len(users[1])))
    print("Number of users in val split: {}".format(len(users[2])))
    print("Number of users in test split: {}".format(len(users[3])))
    print("Number of files: {}".format(n_files))


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("USAGE: python explorer_dataset.py ....")
        exit(1)
    # get_dataset_stats()
    get_specgram_stats()
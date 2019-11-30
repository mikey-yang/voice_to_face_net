"""
Script for training a given model architecture
v0.1 November 29, 2019
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt

# TODO manually split train, validate, IDs and put them into folders


ALPHA = 0
ORTHOGONALIZE_B = False
NUM_EPOCHS = 100
BATCH_SIZE = 10
voice_loss = nn.MSELoss()
face_loss = nn.MSELoss()
LEARNING_RATE = 1e-3
CUDA = False
voice_train_path = ""
voice_validate_path = ""
face_file_format = "data/toy_dataset/facespecs/face_{}.csv"
FACE_SHAPE = None # None == face images are square; else enter a shape tuple


# EDIT THIS CLASS
# class full_model(nn.Module):
#     def __init__(self, w_length, face_length):
#         """
#         w_length: the length of the bottleneck vector i.e. # of basis faces used
#         face_length: the height * width of the face images
#         """
#         super(full_model, self).__init__()
#         self.encoder = nn.ModuleList(
#             [
#                 nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), #1025 x 251
#                 nn.ReLU(True),
#                 #nn.MaxPool2d(2, stride=2), 
#                 nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0), #512 x 125
#                 nn.ReLU(True),
#                 #nn.MaxPool2d(2, stride=2),
#                 nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #256 x 63
#                 nn.ReLU(True),
#                 nn.MaxPool2d(2, stride=2),                             #128 x 31
#                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)   #128 x 31
#             ]
#         )
        
#         self.decoder = nn.ModuleList(
#             [
#                 nn.ConvTranspose2d(1, 32, kernel_size=3, stride=1, padding=1), # 128 x 31
#                 nn.ReLU(True),
#                 nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=(1,0), output_padding = (1,0)), # 256 x 63
#                 nn.ReLU(True),
#                 nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=(1,1), output_padding = (1,0)), # 512 x 125
#                 nn.ReLU(True),
#                 nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=0, output_padding = 0), # 1025 x 251
#                 nn.Tanh()
#             ]
#         )
        
#         self.w_length = w_length
#         self.face_length = face_length
#         self.B = nn.Linear(self.w_length, self.face_length, bias=False)

#     def forward(self, v):
#         # start encoder
#         for layer in self.encoder:
#             v = layer.forward(v)
#             #print(v.shape)
        
#         # collapse final feature map into a vector by taking average across time
#         N, _, H, _ = v.shape
#         w = v.mean(dim=3)
#         w = w.view(N, H)
        
#         # start decoder
#         for layer in self.decoder:
#             v = layer.forward(v)
#             #print(v.shape)
            
#         # face construction
#         f = self.B(w)
        
#         return v, f




def main():
    # import voice data
    train_voice_filenames = get_filenames(voice_train_path)
    validate_voice_filenames = get_filenames(voice_validate_path)
    train_dataset = voice_face(train_voice_filenames, standardize=True)
    validate_dataset = voice_face(validate_voice_filenames, standardize=True)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # import face data as vectors into a dictionary
    train_IDs = set(train_dataset.y.unique()) # type tensor
    validate_IDs = set(validate_dataset.y.unique())
    IDs = train_IDs.union(validate_IDs)
    face_dict = make_face_dict(IDs, path=face_file_format)

    # train model and save outputs
    model = 
    train_model(model, dataloader, face_dict)

def make_face_dict(IDs, path=face_file_format):
    """
    INPUTS:
    - IDs: a 1D tensor or iterable of int IDs
    - path: a filename format with {} in place of the ID in the filename.
    """
    # usage: face_dict[6] returns a 2D numpy array of the face with ID=6
    face_dict = {}
    for ID in IDs:
        if type(ID) == torch.Tensor:
            ID = ID.item()
        assert(type(ID) == int)
        
        face_filename = face_file_format.format(ID)
        face_dict[ID] = np.loadtxt(face_filename, delimiter=',').flatten()
    return face_dict


def get_filenames(voice_paths):
    # in case only one path given, make it a list so that it's iterable
    if type(voice_paths) == str:
        voice_paths = [voice_paths]

    # get lists of all voice filenames
    voice_filenames = []
    for path in voice_paths:
        if path[-1] != '/':
            path += '/'
        voice_filenames += glob(path+"voice_*")
   
    return voice_filenames


class voice_face(Dataset):
    def __init__(self, voice_filenames, standardize=False):
        """
        Preconditions: csv files must contain matrices of the same dimension
        Args:
            voice_filenames (string or list): list of filenames/pathnames of csv files with spectrogram matrices
                                              assumes format voice_{n}_{m}.csv, 
                                              where n is the data ID and m is the spectrogram number for that speaker
            standardise (boolean):            whether to standardize the spectrograms
        """
        # ensure inputs are lists
        if type(voice_filenames) == str:
            voice_filenames = [voice_filenames]
        assert(type(voice_filenames) == list)
                
        # load voice spectrograms one by one
        face_IDs = [] # the face IDs associated with each spectrogram
        matrices = [] # the spectrograms
        for v_file in voice_filenames:
            # get n, the data ID 
            n, _ = get_n_m(v_file)
            face_IDs.append(n)
            
            # get spectrogram
            matrix = np.loadtxt(v_file, delimiter=',', dtype=np.float32)
            if standardize:
                matrix = (matrix - np.mean(matrix)) / np.std(matrix)
            matrices.append(matrix)
        
        # construct spectrograms tensor
        self.X = torch.Tensor(matrices)
        N, D, M = self.X.shape
        self.X = self.X.view(N, 1, D, M) # insert channel dimension
        
        # construct face_IDs tensor
        self.y = torch.tensor(face_IDs)
        
        assert(self.X.shape[0] == self.y.shape[0])
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_n_m(v_file):
    """
    Takes a voice file name of the form path/voice_{n}_{m}.csv
    And outputs n and m as integers.
    """
    v_file = v_file.split('/')[-1] # strip the pathname if it exists
    v_file, _ = v_file.split('.') # strip the file extension
    _, n, m = v_file.split('_') # get n and m from the filename
    return int(n), int(m)

def combined_loss(model_output, labels, face_dict):
    """
    REQUIRES
    - face dictionary
    - face_retrieve_loss function
    INPUTS
    - model_output: a tuple of (voice_output, face_output), both tensors
    - labels: a tuple of (voice_data, face_id), a tensor and a 1-D tensor
    """
    # unpack input
    voice_outputs, face_outputs = model_output
    voice_data, IDs = labels
    
    # compute combined loss
    combined_loss = face_retrieve_loss(face_outputs, IDs, face_dict) + ALPHA * voice_loss(voice_outputs, voice_data)
    return combined_loss


def face_retrieve_loss(face_outputs, IDs, face_dict):
    """
    Retrieves ground truth face tensor given the IDs, computes and returns loss
    REQUIRES
    - face dictionary
    - face_loss function
    INPUTS
    - face_outputs: tensor of face matrices
    - IDs: tensor of IDs
    """
    # construct true faces tensor
    true_faces = []
    for ID in IDs:
        ID_int = ID.item()
        true_faces.append(face_dict[ID_int])
    true_faces = torch.Tensor(true_faces)

    # compute and return loss
    loss = face_loss(face_outputs, true_faces)
    return loss

def train_model(model, dataloader, face_dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    loss_epochs = []
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            # ===================forward=====================
            voice_data, IDs = batch
            # voice_outputs, face_outputs = model(voice)
            loss = combined_loss(model(voice_data), batch, face_dict)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ORTHOGONALIZE_B:
                model.B.weight.data = gram_schmidt(model.B.weight.data)
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, completed at {}'
            .format(epoch+1, NUM_EPOCHS, loss.data.item(), datetime.now()))
        loss_epochs.append(loss)

    save_state("./model_state.pth", model, optimizer, loss_epochs)
    np.savetxt("./convergence/loss.csv", loss_epochs, delimiter=',')
    plt.plot(range(1,len(loss_epochs)+1), loss_epochs)
    plt.title("Convergence of loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./convergence/convergence.png")

def save_state(path, model, optimizer, loss): # epoch, loss
    torch.save({
            'model': str(model),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'epoch': epoch,
            'loss': loss
            }, path)

def load_state(path, model, optimizer, print_model=True): # epoch, loss
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if(print_model == True):
        model_state = checkpoint['model']
        print(model_state)
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

def view_state(model, optimizer, state_size = 0):
    if(state_size == 1):
        print("Model -",model)
        return
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1) # debugged from original repo
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone() # copy first column
    for k in range(1, nk):
        vk = vv[:, k].clone() # debugged from original repo
        uk = 0
        for j in range(0, k): # project vk onto space spanned by bases so far
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu
# source: https://github.com/legendongary/pytorch-gram-schmidt

# helper routine
def conv_shape(L, K, S, P):
    return (L + 2*P - K) // S + 1


main()
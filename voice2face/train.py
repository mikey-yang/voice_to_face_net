import os
import sys
import time
import random 
import math 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import save, dist, load
from itertools import chain
from networks import VGGVoxWrapper, Dictionary, VGGVox
from dataloader import VoxCelebVGGFace
from utils_v2f import Logger
from PIL import Image 
import wandb
wandb.init(project="v2f")

# TRAINING HYPERPARAMETERS
EPOCHS = 100000
BATCHSIZE = 2
LEARNING_RATE = 0.001
NUM_WORKERS = 64
RANDOM_SEED = 15213
# WHERE TO WRITE MODELS
OUTDIRPATH = "models"
LOGGER = None
ALPHA = 1
BETA = 1
# VGGVOX_WEIGHTS = "/share/workhorse3/mahmoudi/voice_to_face_net/src/saved/models/Voice2Face_SpeakerID_VGGVox/0213_120446/model_best.pth"
VGGVOX_WEIGHTS = "/share/workhorse3/mahmoudi/voice_to_face_net/speaker_id_weights.pth"

# SET RANDOM SEEDS
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def val_model(model, n_epoch, dataloader):
    global LOGGER
    model.eval()
    total = 0.0
    correct = 0.0
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    avgloss = 0.0
    iters = 0
    for index, (x,y) in enumerate(dataloader):
        iters += 1
        # y = y.cuda()    
        # print("X: {}".format(x))
        # print("X SHAPE: {}".format(x.size()))
        y_hat = model(x)
        y_hat = y_hat.cpu()
        loss = criterion(y_hat, y)     
        # y_hat = softmax(y_hat)
        # y_hat = y_hat.cpu()
        # y = y.cpu()
        val, index1 = y_hat.max(1)
        # print("Softmax: {}".format(y_hat))
        # print("Labels: {}".format(y))
        correct += ((y-index1) == 0).sum(dim=0).item()

        total += len(y)
        avgloss += loss

    avgloss /= iters
    accuracy = correct/total
    model.train()
    LOGGER.log_accuracy("validation", n_epoch, accuracy, avgloss)
    return avgloss, accuracy

# source: https://github.com/legendongary/pytorch-gram-schmidt
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


def run_epoch(n_epoch, networks, dataloader, optimizer, epoch):
    global LOGGER

    cross_entropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    iters = 0.0
    total_loss = 0.0
    total_spkr_id = 0 
    total_face_recon = 0 

    start_time = time.time()

    voice_network, face_network = networks

    dup_face = None
    dup_gen = None 

    for index, (utt, face, y) in enumerate(dataloader):
        optimizer.zero_grad()

        # update number of iterations 
        iters += 1 

        # get the data loading time 
        end_time = time.time()
   
        # feed the audio to get the embedding 
        embedding = voice_network(utt.float().cuda())
        # generate the face 
        gen_face = face_network(embedding)
        
        # take a copy of the orignal face 
        dup_face = face
        # take a copy of the generated face 
        dup_gen = gen_face

        # calculate the loss 
        y = y.cuda()
        logits = voice_network(embedding, loss=True)
        loss_speakerid = cross_entropy(logits, y)
        loss_face_recon = mse(gen_face, face.float().cuda())

        total_spkr_id += loss_speakerid.item()
        total_face_recon += loss_face_recon.item()

        # combine the loss 
        loss = ALPHA * loss_speakerid + BETA * loss_face_recon

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Orthogalize the face embeddings 
        face_network.linear.weight.data = gram_schmidt(face_network.linear.weight.data)

        all_losses = {"LOSS": loss.item(), 
                      "SPEAKER ID LOSS": loss_speakerid.item(), 
                      "FACE RECONSTRUCTION": loss_face_recon.item()}
        LOGGER.log_minibatch(index, n_epoch, all_losses)
        all_losses["batch"] = index 
        wandb.log(all_losses)

    
    # code to output generated faces during training 
    dup_face = dup_face.detach().cpu().numpy()
    dup_gen = dup_gen.detach().cpu().numpy()
    dup_face_img = Image.fromarray(dup_face[1,:].reshape(128, 128).astype(np.uint8))
    dup_gen_img = Image.fromarray(dup_gen[1,:].reshape(128, 128).astype(np.uint8))

    end_time = time.time()
    avgloss = total_loss/iters #@FIXME: iters --> can be updated to len(dataloader)
    avgloss_speaker_id = total_spkr_id/iters
    avgloss_face = total_face_recon/iters

    LOGGER.log_epoch(n_epoch, "train", {"EPOCH LOSS": avgloss, "SPEAKER ID LOSS": avgloss_speaker_id, "FACE ID LOSS": avgloss_face})

    wandb.log({"epoch": n_epoch+1, "loss": avgloss, "original_face": [wandb.Image(dup_face_img)], "reconstructed_face": [wandb.Image(dup_gen_img)]})

    accuracy = 0
    return avgloss, accuracy

def train(train_dataset):
    global LOGGER

    # init the datasets & data loaders 
    dataset = VoxCelebVGGFace(train_dataset, ["train"])
    data_loader = DataLoader(dataset, BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # init the testing dataset & data loader
    dataset_test = VoxCelebVGGFace(train_dataset, ["test"])
    data_loader_test = DataLoader(dataset, BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # init the model
    voice_network = VGGVoxWrapper(257, 128).cuda()
    face_network = Dictionary(128, 128*128).cuda()

    voice_network.load_state_dict(load(VGGVOX_WEIGHTS))

    optimizer = optim.Adam(chain(voice_network.parameters(), face_network.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)


    # init the logger
    config = {"epochs": EPOCHS, "lr": LEARNING_RATE, "batch_size": BATCHSIZE,
              "random_seed": RANDOM_SEED, "dataset_size": len(dataset)}


    LOGGER = Logger(OUTDIRPATH, config, {"VOICE_NETWORK": voice_network, "FACE_NETWORK": face_network})

    config["timestamp"] = LOGGER.current_timestamp
    config["alpha"] = ALPHA
    config["beta"]  = BETA
    wandb.config.update(config)

    networks = [voice_network, face_network]

    for epoch in range(EPOCHS):
        print("Epoch: {}".format(epoch+1))
        # train an epoch 
        epoch_loss, epoch_acc = run_epoch(epoch, networks, data_loader, optimizer, epoch)
        #@TODO: implement validation model 
        # validate the model 
        # val_loss, epoch_acc = val_model(net, epoch, val_data_loader)
        # check point 
        LOGGER.checkpoint(epoch)
        # write out the logs 
        LOGGER.write_logs()
        # poke your scheduler if you wish...
        # scheduler.step(epoch_loss)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py dataset_mapping")
        exit(1)
    train(sys.argv[1])
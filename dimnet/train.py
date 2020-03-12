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

#MODEL PARAMS
EMBED_DIM = 128
NUM_CLASSES = 1251
HIDDEN_DIMS = [256, 512, 1024]
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


def run_epoch(n_epoch, encoder, classifier, dataloader, optimizer, epoch):
    global LOGGER

    cross_entropy = nn.CrossEntropyLoss()

    iters = 0.0
    total_loss = 0.0

    start_time = time.time()

    for index, (data, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        data, labels = data.cuda(), labels.cuda()
        # update number of iterations 
        # iters += 1 

        # get the data loading time 
        end_time = time.time()
   
        # feed the data to get the embedding 
        embedding = encoder(data.cuda()) #.float()
        # classify
        person_id = classifier(embedding)
        
        # calculate the loss 
        loss = cross_entropy(person_id, labels)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        all_losses = {"LOSS": loss.item()}
        LOGGER.log_minibatch(index, n_epoch, all_losses)
        all_losses["batch"] = index 
        wandb.log(all_losses)

    end_time = time.time()
    avgloss = total_loss/len(dataloader) #UPDATE: iters --> updated to len(dataloader)
    
    LOGGER.log_epoch(n_epoch, "train", {"EPOCH LOSS": avgloss})

    wandb.log({"epoch": n_epoch+1, "loss": avgloss})

    return avgloss


def train(train_dataset, model_mode):
    global LOGGER

    # init the datasets & data loaders 
    dataset = VoxCelebVGGFace(train_dataset, ["train"], model_mode)
    data_loader = DataLoader(dataset, BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # init the testing dataset & data loader
    dataset_test = VoxCelebVGGFace(train_dataset, ["test"], model_mode)
    data_loader_test = DataLoader(dataset, BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # init the model
    encoder_network = None
    if (model_mode == "voice"):
        encoder_network = VGGVoxWrapper(257, EMBED_DIM).cuda()
    if (model_mode == "face"):
        encoder_network = ResNet18().cuda() #@TODO: add the init args
    if (model_mode == "dual"):
        pass
    mlp_network = CommonMLP(EMBED_DIM, HIDDEN_DIMS, NUM_CLASSES)

    #network.load_state_dict(load(VGGVOX_WEIGHTS))

    optimizer = optim.Adam(chain(encoder_network.parameters(),mlp_network.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # init the logger
    config = {"epochs": EPOCHS, "lr": LEARNING_RATE, "batch_size": BATCHSIZE,
              "random_seed": RANDOM_SEED, "dataset_size": len(dataset)}


    LOGGER = Logger(OUTDIRPATH, config, {"ENCODER_NETWORK": encoder_network, "MLP_NETWORK": mlp_network})

    config["timestamp"] = LOGGER.current_timestamp
    config["alpha"] = ALPHA
    config["beta"]  = BETA
    wandb.config.update(config)

    for epoch in range(EPOCHS):
        print("Epoch: {}".format(epoch+1))
        # train an epoch 
        epoch_loss = run_epoch(epoch, encoder_network, mlp_network, data_loader, optimizer, epoch)
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
        print("Usage: python train.py dataset_mapping \"voice\"|\"face\"|\"dual\">")
        exit(1)
    train(sys.argv[1], sys.argv[2])
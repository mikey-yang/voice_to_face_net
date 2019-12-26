import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import torch
from datetime import datetime

def evaluate_model(model, evaluate_IDs, voice_eval_path, face_dict, lineup_length=10, top_n=10, save=True):
    """
    Evaluates the face predictions for a given model with a line up task.
    The task is repeated for every ID given in the evaluate_IDs list, and a list
    of the rank of the real_face in each task is returned. See the function 
    lineup for details of the task.
    INPUTS
    - model:            a given model with a model.predict method that takes an 
                        ID and voice pathname and returns a face reconstruction
    - evaluate_IDs:     a list of IDs to evaluate
    - voice_eval_path:       the pathname where the voice spectrograms of the given 
                        evaluate_IDs can be found, in csv form
    - face_dict:        a dictionary of faces, where the key is the ID as an int
                        and the value is a 2D numpy matrix of the face pixels
    - lineup_length:    the number of IDs to place in the lineup for the task
    """
    if type(evaluate_IDs) == torch.Tensor:
        evaluate_IDs = evaluate_IDs.tolist()
    
    N = len(evaluate_IDs)

    ranks = []
    for i, ID in enumerate(evaluate_IDs):
        face_reconstr = model.predict(ID, voice_eval_path)
        rank, error = lineup(face_reconstr, ID, face_dict, lineup_length=lineup_length)
        ranks.append(rank)
        print("Evaluation number {} of {}: ID={} was rank {}/{}. {}".format(
            i+1, N, ID, rank, lineup_length, datetime.now())
        )

    top_n_acc = top_n_accuracy(ranks, top_n)
    if save:
        np.savetxt("./ranks.csv", ranks, delimiter=',')
        np.savetxt("./top_n_accuracy.csv", top_n_acc, delimiter=',')
    return top_n_acc

def plot_top_n_acc(top_n_acc, save=True):
    n_range = range(1, len(top_n_acc)+1)
    plt.plot(n_range, top_n_acc)
    plt.title("Model Top-n Accuracy on Evaluation Dataset")
    plt.ylabel("Top-n Accuracy")
    plt.xlabel("n")
    plt.xticks(n_range)
    if save:
        plt.savefig("./top_n_accuracy.png")


def top_n_accuracy(ranks, n):
    """
    INPUTS
    - ranks:    1D numpy array of ranks
    - n:        the largest top_n to output
    OUTPUTS
    - np.array  an array of top_n_accuracy where the n'th element is the top_n_accuracy
    """
    if type(ranks) == list:
        ranks = np.array(ranks)
    lineup_length = ranks.size
    n = min(n, lineup_length) # check input isn't erroneous
    
    top_n_accuracy = []
    for i in range(1, n+1):
        top_i_accuracy = np.where(ranks <= i, 1, 0).sum() / lineup_length
        top_n_accuracy.append(top_i_accuracy)
    return np.array(top_n_accuracy)

def lineup(face_reconstr, real_id, face_dict, lineup_length=10):
    """
    Note: includes the training faces in the lineup pool that is sampled from
    """
    assert(lineup_length <= len(face_dict))

    # Set up array of face_ids for the lineup
    other_IDs = list(face_dict.keys())
    other_IDs.remove(real_id) # all IDs except real_id
    others = np.random.choice(other_IDs, size=lineup_length-1, replace=False) # randomly sample
    line_up = np.concatenate(([real_id], others)) # real_id is line_up[0]

    # Calculate the FID between each real face and the reconstructed face
    errors = np.zeros(lineup_length)
    for i in range(lineup_length):
        ID = line_up[i]
        errors[i] = fid(face_reconstr, face_dict[ID])
    
    # minimum FID means the reconstructed face is closest to real face
    order = np.argsort(errors)
    #change from 0-indexing to 1-indexing
    rank = order[0]+1
    return rank, errors[0]


def fid(act1, act2):
    """
    calculate frechet inception distance
    """
    # calculate mean and covariance statistics
    mu1 = act1.mean(axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    mu2 = act2.mean(axis=0)
    sigma2 = np.cov(act2, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
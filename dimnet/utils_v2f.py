import os
from datetime import datetime
from torch import save
from math import sqrt, floor
from collections import defaultdict
import json 
import pickle 


class Logger:
    """
        ADD DESCRIPTION

        model_dir: output directory (writes out the logs and model weights)
        config = {
            "epochs":...,
            "lr":...,
            "batch_size":...,
            "random_seed":...,
            "dataset_size":...
        }
        networks: ["network1":..., "network2":...]
    """

    def __init__(self, model_dir, config, networks):
        self.models = networks
        self.config = config
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.random_seed = config["random_seed"]
        self.dataset_size = config["dataset_size"]
        self.log_rate = floor(sqrt(self.batch_size))

        self.logs = defaultdict(lambda: len(self.logs))

        # create out directory
        self.current_timestamp = self._get_timestamp()
        
        self.outdir = os.path.join(model_dir, self._get_timestamp())
        self._create_folders(self.outdir)

        # write model parameters to a file 
        self._write_model_summary()

    def log_minibatch(self, n_iter, n_epoch, losses):
        """
            logs information after running a minibatch using a log rate

            n_iter: iteration number 
            n_epoch: epoch number 
            losses: dictionary containing the losses {"LOSS_1":...,"LOSS_2:...} 
        """
        if n_iter % self.log_rate != 0:
            return
        n_files = (n_iter+1)*self.batch_size
        files_done = n_files/self.dataset_size

        formatted_losses = "   ".join(["{}:{:.6f}".format(k, v) for k,v in losses.items()])
        print("EPOCH: {}   [{:.2f}% {}/{}]   {}".format(n_epoch+1, files_done*100, 
                                                          n_files, self.dataset_size, formatted_losses))
    
    def log_epoch(self, n_epoch, mode, data):
        """
            Logs after an epoch to sysout & eventually log file 
            n_epoch: epoch number 
            mode: takes "train" or "val"
            data: data to log 
        """
        print("EPOCH: {}".format(n_epoch+1))
        print("MODE: {}".format(mode))
        for k,v in data.items():
            if type(v) == float:
                print("{}: {:.6f}".format(k,v))
            else:
                print("{}: {}".format(k,v))

        self.logs[n_epoch+1] = {mode: data}

    def write_logs(self):
        """
            Writes training stats (after each epoch) to a file in JSON
        """
        f = open(os.path.join(self.outdir, "logs.json"), "w")
        json.dump(self.logs, f, sort_keys=True, indent=4)
        f.close()

    def _get_timestamp(self):
        timestamp = datetime.now()
        return timestamp.strftime("%d-%m-%Y--%H-%M-%S")

    def _create_folders(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _write_model_summary(self):
        f = open(os.path.join(self.outdir, "model_summary.log"), "w")
        for net_name, net in self.models.items():
            f.write(net_name+"\n")
            f.write(str(net)+"\n\n\n")

        f.write("Training Parameters\n")
        for c,v in self.config.items():
            f.write("{}: {}\n".format(c,v))
        f.close()

    def checkpoint(self, n_epoch):
        """
            Checkpoints the models (self.models)

            n_epoch: number of epoch
        """

        for k,v in self.models.items():
            outpath = os.path.join(self.outdir, "{}_epoch_{}.weights".format(k, n_epoch+1))
            save(v.state_dict(), outpath)

         
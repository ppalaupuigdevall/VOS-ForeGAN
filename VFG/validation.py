import time
from options.test_options import TestOptions
from data.dataset_davis import DavisDataset, ValDavisDataset
from models.models import ModelsFactory
from collections import OrderedDict
import os
import torch.utils.data as data
import torch.nn as nn
from utils.visualizer import Visualizer, Visualizerv8
from utils.metrics_utils import db_eval_iou, db_eval_boundary
from natsort import natsorted, ns
import pandas as pd

class Validate:
    def __init__(self):
        self._opt = TestOptions().parse()

        self._dataset_train = ValDavisDataset(self._opt, self._opt.T, self._opt.test_dir)
        self._data_loader_train = data.DataLoader(self._dataset_train, self._opt.batch_size, shuffle=False,num_workers=5)
        self._dataset_train_size = len(self._dataset_train)
        print('# Train videos = %d' % self._dataset_train_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._l1_criterion = nn.L1Loss()
        self._metrics = []
        self._training_T = 8
        self._validate()
        # metrics is a list of dictionaries that will define a DataFrame
                
    def _validation(self):
        J_val_set = 0.0
        Boundary = 0.0
        L1 = 0.0
        for i_val_batch, val_batch in enumerate(self._data_loader_train):
            j_batch, boundary_batch, l1_batch = 0.0, 0.0, 0.0            
            self._model.set_input(val_batch)
            gt_masks = val_batch["masks_"]
            fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
            for t in self._opt.T-1:
                j_batch = j_batch + db_eval_iou(val_batch['gt_masks'][t+1], tensor2im(masks[t]))
                boundary_batch = boundary_batch + db_eval_boundary(tensor2im(masks[t]), val_batch['gt_masks'][t+1])
                # compute l1
                if(t<self._training_T-1):
                    l1_batch = l1_batch + self._l1_criterion(val_batch['imgs'][t+1], fakes[t]).item()
            J_val_set = J_val_set + j_batch/(self._opt.T - 1)
            Boundary = Boundary + boundary_batch/(self._opt.T - 1)
            L1 = l1_batch + l1_batch/(self._training_T-1)
        
        metric = {'iteracio':iteracio, 'j':J_val_set, 'b':Boundary, 'l1':L1}
        return metric

    def _validate(self):
        def what_is_what(s):
            """X_epoch_Y_id_Z.pth, X:net/opt, Y:int, Z:Gf/Gb/Df/Db"""
            l = s.split('.')[0].split('_')
            return l[0],l[2],l[4]

        epochs = []     
        for filename in natsorted(os.listdir(os.path.join(self._opt.save_path, self._opt.name))):
            if '.pth' in filename:
                X,Y,Z = what_is_what(filename)
                if( ('net' in X) and ('Gf' in Z) ):
                    epochs.append(Y)
        for e in epochs:
            print("New Epoch")
            self._model.load_val(e)
            m = self._validation()
            self._metrics.append(m)
            df = pd.DataFrame(self._metrics)
            pd.to_csv(os.path.join(self._opt.save_path, self._opt.name, 'metrics.csv'))
            

if __name__ == "__main__":
    Validate()

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
from data.dataset_davis import tensor2im, resize_img_cv2
import numpy as np
import cv2
import torch
class Validate:
    def __init__(self):
        self._opt = TestOptions().parse()

        self._dataset_train = ValDavisDataset(self._opt, self._opt.T, self._opt.test_dir)
        self._data_loader_train = data.DataLoader(self._dataset_train, self._opt.batch_size, shuffle=False,num_workers=1)
        self._dataset_train_size = len(self._dataset_train)
        print('# Train videos = %d' % self._dataset_train_size)
         
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._l1_criterion = nn.L1Loss()
        self._metrics = []
        self._training_T = 8
        self._validate_batch()
        # self._save_masks()
        # metrics is a list of dictionaries that will define a DataFrame
                
    def _validation(self,iteracio):
        J_val_set = 0.0
        Boundary = 0.0
        L1 = 0.0
        for i_val_batch, val_batch in enumerate(self._data_loader_train):
            j_batch, boundary_batch, l1_batch = 0.0, 0.0, 0.0            
            self._model.set_input(val_batch)
            fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
            for t in range(self._opt.T-1):
                bin_mask = self.binarize_mask(tensor2im(masks[t]), 210)
                j_batch = j_batch + db_eval_iou(tensor2im(val_batch['gt_masks'][t+1]), bin_mask)
                # boundary_batch = boundary_batch + db_eval_boundary(bin_mask, tensor2im(val_batch['gt_masks'][t+1]))
                # compute l1
                if(t<self._training_T-1):
                    l1_batch = l1_batch + self._l1_criterion(val_batch['imgs'][t+1].cuda(), fakes[t]).item()
            J_val_set = J_val_set + j_batch/(self._opt.T - 1)
            # Boundary = Boundary + boundary_batch/(self._opt.T - 1)
            L1 = l1_batch + l1_batch/(self._training_T-1)
        
        metric = {'iteracio':iteracio, 'j':J_val_set/self._dataset_train_size, 'b':Boundary/self._dataset_train_size, 'l1':L1/self._dataset_train_size}
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
            print("New Epoch ", str(e))
            self._model.load_val(str(e))
            m = self._validation(e)
            print("metrics: ", m)
            self._metrics.append(m)
            df = pd.DataFrame(self._metrics)
            df.to_csv(os.path.join(self._opt.save_path, self._opt.name, 'metrics.csv'))
            

    def _validation_batch(self,iteracio):
        
        period = 4 # Evaluation until 4T
        J_val_set = np.zeros((period))
        Boundary = np.zeros((period))
        L1_fg = np.zeros((period))
        L1 = 0.0
        
        for i_val_batch, val_batch in enumerate(self._data_loader_train):
            print(i_val_batch/self._dataset_train_size)
            if len(val_batch['gt_masks'])>=self._opt.T:
                
                j_batch, boundary_batch, l1_batch, l1_fg = np.zeros((period)), np.zeros((period)), np.zeros((period)), np.zeros((period))
                
                self._model.set_input(val_batch)
                fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
                
                for t in range(self._opt.T-period):
                    # for thr in [190,200,210]:

                    bin_mask = self.binarize_mask(tensor2im(masks[t],unnormalize=False), 200)
                    # Jaccard Index
                    j_batch[t//(self._training_T-1)] = j_batch[t//(self._training_T-1)] + db_eval_iou(tensor2im(val_batch['gt_masks'][t+1], unnormalize=False), bin_mask)
                    # L1 fg
                    diff_fgs = self._l1_criterion(val_batch['imgs'][t+1].cuda() * val_batch['gt_masks'][t+1].cuda(), fgs[t]*masks[t])
                    l1_fg[t//(self._training_T-1)] = l1_fg[t//(self._training_T-1)] + diff_fgs.cpu().item()
                    if t<self._training_T-1:
                        l1_batch = l1_batch + self._l1_criterion(val_batch['imgs'][t+1].cuda(), fakes[t]).item()
                
                J_val_set = J_val_set + j_batch/(self._training_T -1)
                Boundary = Boundary + boundary_batch/(self._training_T -1)
                L1 = l1_batch + l1_batch/(self._training_T-1)
                L1_fg = L1_fg + l1_fg/(self._training_T-1)
            
        mets = []
        measures_names = ['j', 'l1', 'l1_fg']
        measures_numeric = [J_val_set, L1, L1_fg]
        for t in range(period):
            to_print = 'T'
            if t>0:    
                to_print = str(t+1) + to_print
            for iidx, me in enumerate(measures_names):
                metric = {}
                metric['iteracio']=iteracio
                metric['timestep']=to_print
                metric[me] = measures_numeric[iidx][t]/(self._dataset_train_size) # -1 because there's one video that has less than 32 frames in training
                mets.append(metric)
        return mets

    def _validation_batch_for_each_t(self,iteracio):
        
        period = 4 # Evaluation until 4T
        n_digits = 4
        mets = [] # mets of all batches
        num_frames_eval = self._opt.T - period
        for i_val_batch, val_batch in enumerate(self._data_loader_train):
            mets_batch = []
            print(i_val_batch/self._dataset_train_size)
            video_name = val_batch["video_name"][0]
            if len(val_batch['gt_masks'])>=self._opt.T:
                j_batch, boundary_batch, l1_batch, l1_fg = np.zeros((6,num_frames_eval)), np.zeros((num_frames_eval)), np.zeros((num_frames_eval)), np.zeros((num_frames_eval))                
                self._model.set_input(val_batch)
                fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
                for t in range(self._opt.T-period):
                    diff_fgs = self._l1_criterion(val_batch['imgs'][t+1].cuda() * val_batch['gt_masks'][t+1].cuda(), fgs[t])
                    l1_fg[t] = np.round(diff_fgs.cpu().item(), n_digits)
                    if t<self._training_T-1:
                        wa = self._l1_criterion(val_batch['imgs'][t+1].cuda(), fakes[t]).item()
                        l1_batch[t] = np.round(wa,n_digits)
                    for i_thr, thr in enumerate([1,128,180,190,200,210]):
                        bin_mask = self.binarize_mask(tensor2im(masks[t],unnormalize=False), thr)
                        wae = db_eval_iou(tensor2im(val_batch['gt_masks'][t+1], unnormalize=False), bin_mask)
                        j_batch[i_thr, t] = np.round(wae, n_digits)
                        # Create row of dataframe 
                        to_print = 'T'
                        current_period = t//(self._training_T-1)    
                        if current_period>0:    
                            to_print = str(current_period) + to_print
                        metric = {}
                        metric['iteracio'] = iteracio
                        metric['video_name'] = video_name
                        metric['t'] = t
                        metric['thr'] = thr
                        metric['J'] = j_batch[i_thr, t]
                        metric['L1_fg'] = l1_fg[t]
                        metric['L1'] = l1_batch[t]
                        metric['period'] = to_print
                        mets_batch.append(metric)
                mets.extend(mets_batch)   
        return mets


    def _validate_batch(self):
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
            print("New Epoch ", str(e))
            self._model.load_val(str(e))
            m = self._validation_batch_for_each_t(e)
            print("metrics: ", m)
            self._metrics.extend(m)
            df = pd.DataFrame(self._metrics)
            df.to_csv(os.path.join(self._opt.save_path, self._opt.name, self._opt.name+"metrics.csv"))
            
    def _save_masks(self):
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
            print("New Epoch ", str(e))
            self._model.load_val(str(e))
            _ = self._inferencia(e)
    
    def _inferencia(self, iteracio):
        period = 4 # Evaluation until 4T
        guagua = iter(self._data_loader_train)
        val_batch = next(guagua)

        self._model.set_input(val_batch)
        fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
        cat = 'stroller'
        print("uepa")
        for t in range(self._opt.T-period):
            print(t)
            if t%(self._training_T-1) == (self._training_T-2):
                
                bin_mask = self.binarize_mask(tensor2im(masks[t]), 190)
                # cv2.imwrite(os.path.join(self._opt.save_path,self._opt.name,'masks','z_mask_'+"{:04d}".format(t)+'_debug_'+ "{:04d}".format(int(iteracio)) + '.jpeg'), bin_mask)
                cv2.imwrite(os.path.join(self._opt.save_path,self._opt.name,'masks','z_GT_mask_'+"{:04d}".format(t)+'_debug_'+ "{:04d}".format(int(iteracio)) + '.jpeg'), tensor2im(val_batch['gt_masks'][t+1], unnormalize=False))
                # Draw contours:
                imgs_noms = self._dataset_train.get_noms()
                img_name = os.path.join(self._opt.img_dir,cat, imgs_noms[t+1])    
                im = resize_img_cv2(cv2.imread(img_name), self._opt.resolution)
                image, contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                im[:, :, 0] = (bin_mask > 0) * 255 + (bin_mask == 0) * im[:, :, 0]
                cnt = contours[0]
                im=cv2.drawContours(im, contours, -1, (0, 0, 0), 1)
                cv2.imwrite(os.path.join(self._opt.save_path,self._opt.name,'masks','a_mask_T' +"{:04d}".format(t)+'_'+ "{:04d}".format(int(iteracio)) + '.jpeg'), im)
        
        return ""


    def binarize_mask(self,maska, th=210): 
        ret, bin_mask = cv2.threshold(maska, th, 255, cv2.THRESH_BINARY)
        bin_mask_3channels = np.zeros((self._opt.resolution[0], self._opt.resolution[1], 3))
        bin_mask_3channels[:,:,0] = bin_mask
        bin_mask_3channels[:,:,1] = bin_mask
        bin_mask_3channels[:,:,2] = bin_mask
        return bin_mask_3channels       
        # return bin_mask

if __name__ == "__main__":
    Validate()

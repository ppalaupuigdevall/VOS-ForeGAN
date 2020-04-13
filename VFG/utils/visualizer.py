import numpy as np
import os
import time
from . import util
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, opt):
        self._opt = opt
        self._T = opt.T
        self._save_path = os.path.join(opt.save_path, opt.name)
        self._writer = SummaryWriter(self._save_path)
        self.counter = 0
    def display_current_results(self,imgs, iteracio):
        for t in range(self._T-1):
            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(imgs['masks'][t][:,:,0])
            self._writer.add_figure('imgs/mask_'+str(self.counter*(self._T -1 )).zfill(6), fig, iteracio)
            # self._writer.add_image('imgs/mask_'+str(self.counter*(self._T -1 )).zfill(6), imgs['masks'][t].reshape(3,224,416), iteracio)
            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(imgs['fgs'][t])
            self._writer.add_figure('imgs/fgs_'+str(self.counter*(self._T -1 )).zfill(6), fig, iteracio)
            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(imgs['bgs'][t])
            self._writer.add_figure('imgs/bgs_'+str(self.counter*(self._T -1 )).zfill(6), fig, iteracio)
            # self._writer.add_image('imgs/bgs_'+str(self.counter*(self._T -1 )).zfill(6), imgs['bgs'][t].reshape(3,224,416), iteracio)
            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(imgs['fakes'][t])
            self._writer.add_figure('imgs/fakes_'+str(self.counter*(self._T -1 )).zfill(6), fig, iteracio)
            # self._writer.add_image('imgs/fakes_'+str(self.counter*(self._T -1)).zfill(6), imgs['fakes'][t].reshape(3,224,416), iteracio)
        self.counter = self.counter + 1
    
    def plot_scalars(self, scalars, iteracio):
        self._writer.add_scalar('losses/L1_loss',scalars['loss_g_fb_rec'], iteracio)
        self._writer.add_scalar('losses/Db',scalars['loss_db'], iteracio)
        self._writer.add_scalar('losses/Df',scalars['loss_df'], iteracio)        
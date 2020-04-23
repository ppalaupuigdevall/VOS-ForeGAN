import time
from options.train_options import TrainOptions
from data.dataset_davis import DavisDataset
from models.models import ModelsFactory
# from utils.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os
import torch.utils.data as data
from utils.visualizer import Visualizer

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()

        self._dataset_train = DavisDataset(self._opt, self._opt.T, self._opt.OF_dir)
        self._data_loader_train = data.DataLoader(self._dataset_train, self._opt.batch_size, drop_last=True, shuffle=True)
        self._dataset_train_size = len(self._dataset_train)
        print('# Train videos = %d' % self._dataset_train_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        
        self._tb_visualizer = Visualizer(self._opt)

        self._train()


    def _train(self):
        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._iteracio = 0
        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            if(i_epoch % 100 == 0):
                self._model.save(i_epoch)
                print('saving the model at the end of epoch %d' % (i_epoch))

            # update learning rate
            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()


    def _train_epoch(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        print("--- - - - - - - EPOCH ", i_epoch, " - - - - ")
        for i_train_batch, train_batch in enumerate(self._data_loader_train):
            iter_start_time = time.time()
            # train model
            self._model.set_input(train_batch)
            train_generator = ((i_train_batch+1) % self._opt.train_G_every_n_iterations == 0)
            self._model.optimize_parameters(train_generator=train_generator)
            self._iteracio = self._iters_per_epoch*i_epoch + i_train_batch
            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size
            # display visualizer
            if i_epoch%10 == 0:
                self._display_visualizer_train(self._iteracio)
                self._last_display_time = time.time()
            
    def _display_visualizer_train(self, iteracio):
        self._tb_visualizer.display_current_results(self._model.get_imgs(), iteracio)
        self._tb_visualizer.plot_scalars(self._model.get_losses(), iteracio)

if __name__ == "__main__":
    Train()

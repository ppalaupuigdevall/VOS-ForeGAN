import time
from options.train_options import TrainOptions
from data.dataset_davis import DavisDataset
from models.models import ModelsFactory
from collections import OrderedDict
import os
import torch.utils.data as data
from utils.visualizer import Visualizer, Visualizerv8
from utils.metrics_utils import db_eval_iou, db_eval_boundary

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()

        self._dataset_train = DavisDataset(self._opt, self._opt.T, self._opt.OF_dir)
        # self._dataset_val = ValDavisDataset(self._opt, self._opt.T, self._opt.OF_dir)
        self._data_loader_train = data.DataLoader(self._dataset_train, self._opt.batch_size, drop_last=True, shuffle=True,num_workers=4)
        self._dataset_train_size = len(self._dataset_train)
        print('# Train videos = %d' % self._dataset_train_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        
        self._tb_visualizer = Visualizer(self._opt)
        # self._tb_visualizer = Visualizerv8(self._opt)

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

            if(i_epoch % 150 == 0):
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

            if i_epoch%self._opt.save_scalars == 0:
                self._display_visualizer_scalars_train(self._iteracio)
                self._last_display_time = time.time()
            
            if i_epoch%self._opt.save_imgs == 0:
                self._display_visualizer_imgs_train(self._iteracio)
                self._last_display_time = time.time()
                
    def _validation(self, iteracio):
        J_val_set = 0.0
        Boundary = 0.0
        for i_val_batch, val_batch in enumerate(self._data_loader_train):
            jbatch = 0.0
            boundary_batch = 0.0
            self._model.set_input(val_batch)
            fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
            
            for t in self._opt.T-1:
                jbatch = jbatch + db_eval_iou(val_batch['gt_mask'][t], tensor2im(masks[t]))
                boundary_batch = boundary_batch + db_eval_boundary(tensor2im(masks[t]), val_batch['gt_mask'][t])
            J_val_set = J_val_set + jbatch/(self._opt.T - 1)
            Boundary = Boundary + boundary_batch/(self._opt.T - 1)
        print("########### VALIDATION in T = %d ############"%(self._opt.T))
        print("Jaccard index J = %.2f"%(J_val_set))
        print("Boundary F score = %.2f"%(Boundary))



    

    def _display_visualizer_scalars_train(self, iteracio):
        self._tb_visualizer.plot_scalars(self._model.get_losses(), iteracio)
    def _display_visualizer_imgs_train(self, iteracio):
        self._tb_visualizer.display_current_results(self._model.get_imgs(), iteracio)

if __name__ == "__main__":
    Train()

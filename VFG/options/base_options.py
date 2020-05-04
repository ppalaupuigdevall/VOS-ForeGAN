import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--img_dir',          type=str,   default="/data/Ponc/DAVIS/JPEGImages/480p/",    help='path to imgs folder')
        self._parser.add_argument('--OF_dir',           type=str,   default="/data/Ponc/DAVIS/OpticalFlows/",       help='path to OFs folder')
        self._parser.add_argument('--mask_dir',         type=str,   default="/data/Ponc/DAVIS/Annotations/480p/",   help='path to masks folder')
        self._parser.add_argument('--resolution',       type=tuple, default=(224, 416),                             help='default image resolution')
        self._parser.add_argument('--T',                type=int,   default=8,                                     help='temporal horizon')
        self._parser.add_argument('--batch_size',       type=int,   default=1,                                      help='input batch size')
        self._parser.add_argument('--gpu_ids',          type=str,   default='0',                                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self._parser.add_argument('--model',            type=str,   default='forestgan_pure_rnn',                   help='there is only this one so do not change it')
        self._parser.add_argument('--load_epoch',       type=int,   default=200,                                   help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--name',             type=str,   default='experiment_v1',                        help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--checkpoints_dir',  type=str,   default='/data/Ponc/VOS-ForeGAN/',              help='models are saved here')
        self._parser.add_argument('--save_path',        type=str,   default='/data/Ponc/VOS-ForeGAN/',              help='TensorboardX directory')

        
        # Foreground/Background patches extractions
        self._parser.add_argument('--num_patches', type=int, default = 10, help='number of patches/image to be compared')
        self._parser.add_argument('--kh', type=int, default = 60, help='number of patches to be compared')
        self._parser.add_argument('--kw', type=int, default = 112, help='number of patches to be compared')
        self._parser.add_argument('--stride_h', type=int, default=3, help='stride patches')
        self._parser.add_argument('--stride_w', type=int, default=3, help='stride patches')

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        # self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        # vars returns a dictionary
        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        # self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)


        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])
            

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')


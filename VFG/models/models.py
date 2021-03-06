import os
import torch
from torch.optim import lr_scheduler

class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None

        if model_name == 'forestgan_convlstm_v0':
            from models.forestgan_convlstm_v0 import ForestGAN_ConvLSTM_v0
            model = ForestGAN_ConvLSTM_v0(*args, **kwargs)
        elif model_name == 'forestgan_rnn_basic_feat':
            from models.forestgan_rnn_basic_feat import ForestGANRNN_basic_feat
            model = ForestGANRNN_basic_feat(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v1':
            from models.forestgan_rnn_v1 import ForestGANRNN_v1
            model = ForestGANRNN_v1(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v1_antic':
            from models.forestgan_rnn_v1_antic import ForestGANRNN_v1_antic
            model = ForestGANRNN_v1_antic(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v3':
            from models.forestgan_rnn_v3 import ForestGANRNN_v3
            model = ForestGANRNN_v3(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v5':
            from models.forestgan_rnn_v5 import ForestGANRNN_v5
            model = ForestGANRNN_v5(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v7':
            from models.forestgan_rnn_v7 import ForestGANRNN_v7
            model = ForestGANRNN_v7(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v9':
            from models.forestgan_rnn_v9 import ForestGANRNN_v9
            model = ForestGANRNN_v9(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v10':
            from models.forestgan_rnn_v10 import ForestGANRNN_v10
            model = ForestGANRNN_v10(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v02':
            from models.forestgan_rnn_v02 import ForestGANRNN_v02
            model = ForestGANRNN_v02(*args, **kwargs)
        elif model_name == 'forestgan_rnn_extended_feat':
            from models.forestgan_rnn_extended_feat import ForestGANRNN_extended_feat
            model = ForestGANRNN_extended_feat(*args, **kwargs)
        elif model_name == 'forestgan_rnn_extended_fg':
            from models.forestgan_rnn_extended_fg import ForestGANRNN_extended_fg
            model = ForestGANRNN_extended_fg(*args, **kwargs)
        elif model_name == 'forestgan_convlstm_basic_feat':
            from models.forestgan_convlstm_basic_feat import ForestGAN_ConvLSTM_basic_feat
            model = ForestGAN_ConvLSTM_basic_feat(*args, **kwargs)
        elif model_name == 'forestgan_convlstm_basic_fg':
            from models.forestgan_convlstm_basic_fg import ForestGAN_ConvLSTM_basic_fg
            model = ForestGAN_ConvLSTM_basic_fg(*args, **kwargs)


        elif model_name == 'forestgan_rnn_noof':

            from models.forestgan_rnn_noof import ForestGANRNN_noof
            model = ForestGANRNN_noof(*args, **kwargs)
        elif model_name == 'train_gb':
            from models.train_only_bg import TrainGB
            model = TrainGB(*args, **kwargs)
        elif model_name == 'forestgan_rnn_v8':
            from models.forestgan_rnn_v8 import ForestGANRNN_v8
            model = ForestGANRNN_v8(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt
        self._gpu_ids = opt.gpu_ids
        self._is_train = opt.is_train
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.name)


    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
       
        load_path = os.path.join(self._save_dir, load_filename)
        # load_path=None
        # load_path = os.path.join('/data/Ponc/VOS-ForeGAN/experiment_13/',load_filename)
        # if((('Gb' in optimizer_label) or ('Db' in optimizer_label))):
        #     load_filename = 'opt_epoch_1050_id_'+str(optimizer_label)+'.pth'
        #     load_path = os.path.join('/data/Ponc/VOS-ForeGAN/experiment_v031/',load_filename)
        assert os.path.exists(load_path), 'Weights file not found. ' % load_path

        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        print(load_path)
        # load_path=None
        # load_path = os.path.join('/data/Ponc/VOS-ForeGAN/experiment_13/',load_filename)
        # if((('Gb' in network_label) or ('Db' in network_label))):
        #     print(network_label)
        #     load_filename = 'net_epoch_1050_id_'+str(network_label)+'.pth'
        #     print(load_filename)
        #     load_path = os.path.join('/data/Ponc/VOS-ForeGAN/experiment_v031/',load_filename)
        #     print(load_path)
        assert os.path.exists(load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        network.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
        print('loaded net: %s' % load_path)

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    def _get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

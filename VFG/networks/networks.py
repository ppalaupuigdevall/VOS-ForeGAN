import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'generator_wasserstein_gan_f':
            from networks.generator_wasserstein_gan import GeneratorF
            network = GeneratorF(*args, **kwargs)

        elif network_name == 'generator_wasserstein_gan_b':
            from networks.generator_wasserstein_gan import GeneratorB
            network = GeneratorB(*args, **kwargs)
            
        elif network_name == 'discriminator_wasserstein_gan':
            from networks.discriminator_wasserstein_gan_patchgan import Discriminator
            network = Discriminator(*args, **kwargs)
        
        elif network_name == 'discriminator_wasserstein_gan_M':
            from networks.discriminator_wasserstein_gan_patchgan import Discriminator_M
            network = Discriminator_M(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_b_static_ACR':
            from networks.generator_wasserstein_gan import GeneratorB_static_ACR
            network = GeneratorB_static_ACR(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_b_static_ACR_OF':
            from networks.generator_wasserstein_gan import GeneratorB_static_ACR_OF
            network = GeneratorB_static_ACR_OF(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_f_static_ACR':
            from networks.generator_wasserstein_gan import GeneratorF_static_ACR
            network = GeneratorF_static_ACR(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_f_static_ACR_mask_from_fg':
            from networks.generator_wasserstein_gan import GeneratorF_static_ACR_mask_from_fg
            network = GeneratorF_static_ACR_mask_from_fg(*args, **kwargs)

        elif network_name == 'generator_wasserstein_gan_b_static_ACR_single_img':
            from networks.generator_wasserstein_gan import GeneratorB_static_ACR_single_img
            network = GeneratorB_static_ACR_single_img(*args, **kwargs)

        elif network_name == 'generator_wasserstein_gan_f_static_ACR_noOF':
            from networks.generator_wasserstein_gan import GeneratorF_static_ACR_noOF
            network = GeneratorF_static_ACR_noOF(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_f_convlstm':
            from networks.convlstm import GeneratorF_convLSTM
            network = GeneratorF_convLSTM(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_f_convlstm_mask_from_fg':
            from networks.convlstm import GeneratorF_convLSTM_mask_from_fg
            network = GeneratorF_convLSTM_mask_from_fg(*args, **kwargs)
        elif network_name == 'generator_wasserstein_gan_b_convlstm':
            from networks.convlstm import GeneratorB_convLSTM 
            network = GeneratorB_convLSTM(*args, **kwargs)        

        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')
        self._parser.add_argument('--print_freq_s', type=int, default=10, help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=360, help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=3600, help='frequency of saving the latest results')

        self._parser.add_argument('--nepochs_no_decay', type=int, default=15000, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=20, help='# of epochs to linearly decay learning rate to zero')

        # Architecture details
        
        self._parser.add_argument('--extra_ch_gf', type=int, default=2, help='number of channels of the input to Gf')
        self._parser.add_argument('--extra_ch_gb', type=int, default=2, help='number of channels of the input to Gb')

        # Optimizers parameters
        self._parser.add_argument('--train_G_every_n_iterations', type=int, default=1, help='train G every n interations')
        self._parser.add_argument('--poses_g_sigma', type=float, default=0.06, help='initial learning rate for adam')
        
        self._parser.add_argument('--lr_Gf', type=float, default=0.0001, help='initial learning rate for Gf adam')
        self._parser.add_argument('--lr_Gb', type=float, default=0.0001, help='initial learning rate for Gb adam')
        self._parser.add_argument('--Gf_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--Gb_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--Gf_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self._parser.add_argument('--Gb_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        
        self._parser.add_argument('--lr_Df', type=float, default=0.0001, help='initial learning rate for D adam')
        self._parser.add_argument('--lr_Db', type=float, default=0.0001, help='initial learning rate for D adam')
        self._parser.add_argument('--Df_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self._parser.add_argument('--Db_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self._parser.add_argument('--Df_adam_b2', type=float, default=0.999, help='beta2 for D adam')
        self._parser.add_argument('--Db_adam_b2', type=float, default=0.999, help='beta2 for D adam')

        # Lambdas for losses
        self._parser.add_argument('--lambda_Df_prob', type=float, default=100, help='lambda for real/fake discriminator loss')
        self._parser.add_argument('--lambda_Db_prob', type=float, default=1, help='lambda for real/fake discriminator loss')
        self._parser.add_argument('--lambda_rec', type=float, default=1, help='lambda reconstruction loss')
        self._parser.add_argument('--lambda_Df_gp', type=float, default=1, help='lambda gradient penalty loss')
        self._parser.add_argument('--lambda_Db_gp', type=float, default=1, help='lambda gradient penalty loss')

        # Foreground/Background patches extractions
        self._parser.add_argument('--num_patches', type=int, default = 40, help='number of patches/image to be compared')
        self._parser.add_argument('--kh', type=int, default = 60, help='number of patches to be compared')
        self._parser.add_argument('--kw', type=int, default = 112, help='number of patches to be compared')
        self._parser.add_argument('--stride_h', type=int, default=10, help='stride patches')
        self._parser.add_argument('--stride_w', type=int, default=12, help='stride patches')

        self.is_train = True

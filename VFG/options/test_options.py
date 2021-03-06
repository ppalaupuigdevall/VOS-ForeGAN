from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--test_dir',          type=str,   default='/data/Ponc/DAVIS/OpticalFlows/validation/')
        self._parser.add_argument('--test_batch_size',   type=int,   default=1)
        self._parser.add_argument('--test_dir_save',     type=str,   default='./imgs/generated/')
        self._parser.add_argument('--init_frame',        type=int,   default=0)
        self._parser.add_argument('--use_moments',       type=bool,   default=False)
        self.is_train = False
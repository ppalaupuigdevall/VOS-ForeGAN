import numpy as np
import os
import time
from . import util
from tensorboardX import SummaryWriter


class Visualizer:
    def __init__(self, opt):
        self._opt = opt

        self._save_path = os.path.join(opt.save_path, opt.name)
        self._writer = SummaryWriter(self._save_path)
    
    
        
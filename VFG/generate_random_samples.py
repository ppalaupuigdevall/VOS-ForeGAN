import time
from options.test_options import TestOptions
from data.dataset_davis import DavisDataset, tensor2im
from models.models import ModelsFactory
import os
import torch.utils.data as data
import cv2

class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._dataset_test = DavisDataset(self._opt, self._opt.T, self._opt.test_dir)
        self._data_loader_test = data.DataLoader(self._dataset_test, self._opt.test_batch_size, drop_last=True, shuffle=True)
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._generate_something()

    def _generate_something(self):
        for i_test_batch, test_batch in enumerate(self._data_loader_test):
            self._model.set_input(test_batch)
            fgs, bgs = self._model.forward(self._opt.T)

        def rgb2bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for t in range(self._opt.T-1):
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'fg_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(fgs[t])))
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'bg_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(bgs[t])))


if __name__ == "__main__":
    Test()
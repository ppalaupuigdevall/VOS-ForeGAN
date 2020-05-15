from options.test_options import TestOptions
from data.dataset_davis import DavisDataset, tensor2im, resize_img_cv2
from models.models import ModelsFactory
import os
import torch.utils.data as data
import cv2
from imutils import build_montages
import numpy as np

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
            fgs, bgs, fakes, masks = self._model.forward(self._opt.T)

        def rgb2bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cat = self._dataset_test.categories[0]
        imgs = self._dataset_test.imgs_by_cat[cat]
        
        for t in range(self._opt.T-1):
            img_name = os.path.join(self._dataset_test.img_dir,cat, imgs[t+1])    
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'fg_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(fgs[t])))
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'bg_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(bgs[t])))
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'fake_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(fakes[t])))
            im = resize_img_cv2(cv2.imread(img_name), self._opt.resolution)
            mask = np.reshape(tensor2im(masks[t]), self._opt.resolution) 
            ret, bin_mask = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_debug_'+ "{:02d}".format(t) + '.jpeg'), bin_mask)
            # Draw contours:
            image, contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            im[:, :, 0] = (bin_mask > 0) * 255 + (bin_mask == 0) * im[:, :, 0]
            cnt = contours[4]
            im=cv2.drawContours(im, contours, -1, (0, 0, 0), 1)
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_'+ "{:02d}".format(t) + '.jpeg'), im)

    


if __name__ == "__main__":
    Test()
from options.test_options import TestOptions
from data.dataset_davis import DavisDataset, tensor2im, resize_img_cv2
from models.models import ModelsFactory
import os
import torch.utils.data as data
import cv2
from imutils import build_montages
import numpy as np
import torch
from SOS.Q import Q_real_M
import pandas as pd

"""
python VFG/generate_random_samples_RGB.py --model forestgan_rnn_v1_antic --name experiment_v1 --load_epoch 4500 --use_moments True
"""
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
            fgs, bgs, fakes, masks,f = self._model.forward(self._opt.T)
            break
        def rgb2bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cat = self._dataset_test.categories[0]
        cat = self._dataset_test._cat
        imgs = self._dataset_test.imgs_by_cat[cat]

        for t in range(self._opt.T-1):
            print(t)
            
            img_name = os.path.join(self._dataset_test.img_dir,cat, imgs[t+1])    
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'fg_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(fgs[t])))
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'bg_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(bgs[t])))
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'fake_'+ "{:02d}".format(t) + '.jpeg'), rgb2bgr(tensor2im(fakes[t])))
            im = resize_img_cv2(cv2.imread(img_name), self._opt.resolution)
            mask = np.reshape(tensor2im(masks[t],unnormalize=False), self._opt.resolution) 
            ret, bin_mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

            bin_mask_discarded = np.zeros((224,416,3),dtype=np.uint8)
            bin_mask_discarded[:,:,0] = bin_mask.copy()
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_before_moments_'+ "{:02d}".format(t) + '.jpeg'), bin_mask)
            if self._opt.use_moments and t==0:
                img_0 = test_batch['imgs'][0][0]
                gt_mask = test_batch['mask'][0,:,:,:]
                mom = Q_real_M(3,3)
                num_examples = 8
                idxs_nz = torch.nonzero(gt_mask[0,:,:])
                num_cops = self._opt.resolution[0]
                img_0 = torch.as_tensor(img_0, dtype=torch.float32) # numpy
                for p in range(idxs_nz.size()[0]):
                    _ = mom(img_0[:,idxs_nz[p][0],idxs_nz[p][1]].view(1,3).cuda()) # torch
                mom.set_build_M()
            
            
            num_examples = 8
            if self._opt.use_moments:
                with torch.no_grad():
                    momento = torch.zeros(224,416)
                    idxs_nz = torch.nonzero(torch.from_numpy(bin_mask[:,:]))
                    variable_auxiliar = torch.zeros(idxs_nz.size()[0],3)
                    for p in range(idxs_nz.size()[0]):
                        img_t = test_batch['imgs'][t+1][0]
                        variable_auxiliar[p,:] = img_t[:,idxs_nz[p][0],idxs_nz[p][1]]# numpy
                    for elem_i in range(int(idxs_nz.size()[0]//num_examples)):
                        elem = variable_auxiliar[elem_i*num_examples:elem_i*num_examples+num_examples,:].unsqueeze(0)                     
                        elem = elem.cuda()
                        wa = mom(elem.view(num_examples,3))
                        predictions_out = torch.ge(wa,10).view(num_examples)
                        for pre_i in range(predictions_out.size()[0]):
                            if predictions_out[pre_i]:
                                bin_mask[idxs_nz[int(elem_i*num_examples + pre_i)][0],idxs_nz[int(elem_i*num_examples+pre_i)][1]] = 0
                                bin_mask_discarded[idxs_nz[int(elem_i*num_examples + pre_i)][0],idxs_nz[int(elem_i*num_examples+pre_i)][1],2] = 255
                            
  
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_discarded_'+ "{:02d}".format(t) + '.jpeg'), bin_mask_discarded)
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_debug_'+ "{:02d}".format(t) + '.jpeg'), bin_mask)
            # Draw contours:
            image, contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            im[:, :, 0] = (bin_mask > 0) * 255 + (bin_mask == 0) * im[:, :, 0]
            cnt = contours[0]
            im=cv2.drawContours(im, contours, -1, (0, 0, 0), 1)
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_'+ "{:02d}".format(t) + '.jpeg'), im)


if __name__ == "__main__":
    Test()
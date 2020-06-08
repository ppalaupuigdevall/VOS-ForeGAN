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
            if self._opt.use_moments:
                fgs, bgs, fakes, masks, features = self._model.forward(self._opt.T)
                feature = features[0]
                gt_mask = test_batch['mask'][0,:,:,:]
                mom = Q_real_M(64,2)
                
                num_examples = 1
                variable_auxiliar = torch.zeros(num_examples,64)
                contador = 0
                idxs_nz = torch.nonzero(gt_mask[0,:,:])
                mets = []
                num_cops = self._opt.resolution[0]
                for a in range(self._opt.resolution[0]):
                    print(a/num_cops)
                    for b in range(self._opt.resolution[1]):
                        met = {}
                        if(gt_mask[0,a,b]>0.0):
                            met['in_out'] = 'in'
                            feat = feature[:,:,a,b]
                            for ua in range(64):
                                met[ua] = feat[0,ua].item()
                            mets.append(met)
                        else:
                            met['in_out'] = 'out'
                            feat = feature[:,:,a,b]
                            for ua in range(64):
                                met[ua] = feat[0,ua].item()
                            mets.append(met)
                df = pd.DataFrame(mets)
                print("Saving csv")
                df.to_csv(os.path.join(self._opt.save_path, self._opt.name, 'features_in_out.csv'))                



                print(idxs_nz.size())
                print(idxs_nz.size()[0]/num_examples)
                for p in range(idxs_nz.size()[0]):
                    print(p/idxs_nz.size()[0])
                    _ = mom(feature[:,:,idxs_nz[p][0],idxs_nz[p][1]])                        
                mom.set_build_M()
            else:
                fgs, bgs, fakes, masks = self._model.forward(self._opt.T)
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
            num_examples = 1
            if self._opt.use_moments:
                with torch.no_grad():
                    momento = torch.zeros(224,416)
                    variable_auxiliar = torch.zeros(num_examples,64)
                    contador = 0
                    idxs_nz = torch.nonzero(torch.from_numpy(bin_mask[:,:]))
                    
                    # idxs_nz es relaciona amb llista_dim_v perque lelement 
                    llista_dim_v = [] # llista amb dim_v - 1 candidats, cada element es torch.tensor(2144,64)
                    for p in range(idxs_nz.size()[0]):
                        if(contador == num_examples):
                            contador = 0
                            llista_dim_v.append(variable_auxiliar)
                            variable_auxiliar = torch.zeros(num_examples,64)
                        elif contador < num_examples:
                            variable_auxiliar[contador,:] = feature[:,:,idxs_nz[p][0],idxs_nz[p][1]]                        
                            contador = contador + 1
                    for elem_i, elem in enumerate(llista_dim_v):
                        elem = elem.cuda()
                        print("UEPA")
                        wa = mom(elem)
                        predictions_out = torch.ge(wa,2145).view(num_examples)
                        for pre_i in range(predictions_out.size()[0]):
                            if predictions_out[pre_i]:
                                bin_mask[idxs_nz[int(elem_i*num_examples + pre_i)][0],idxs_nz[int(elem_i*num_examples+pre_i)][1]] = 0

   

            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_debug_'+ "{:02d}".format(t) + '.jpeg'), bin_mask)
            # Draw contours:
            image, contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            im[:, :, 0] = (bin_mask > 0) * 255 + (bin_mask == 0) * im[:, :, 0]
            cnt = contours[0]
            im=cv2.drawContours(im, contours, -1, (0, 0, 0), 1)
            cv2.imwrite(os.path.join(self._opt.test_dir_save,'mask_'+ "{:02d}".format(t) + '.jpeg'), im)


    


if __name__ == "__main__":
    Test()
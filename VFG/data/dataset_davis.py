import os
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
import json
from utils_f.utils_flow import warp_flow
from utils_f.utils_flow import draw_flow
from utils_f.utils_flow import readFlow
import math
import torch

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)



def resize_img(img, size):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    return img_pil_resized

def resize_gray_img(img, size):
    img_pil = Image.fromarray(img)
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    return img_pil_resized

def resize_img_cv2(img, size):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    img_cv2_resized = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2BGR)
    return img_cv2_resized


def remap_values(values, xmin, xmax, ymin, ymax):
    """
    Remaps values with a positive straight line (es podria posar com a parametre si fos ppendent >0 o <0)
    """
    values = np.clip(values, xmin, xmax)
    m = (ymax - ymin)/(xmax - xmin)
    n = m * xmin
    values = np.uint8(m * values + n)
    return values


class DavisDataset(data.Dataset):
    
    def __init__(self, conf, T, OF_dir):
        super(DavisDataset, self).__init__()
        
        self.img_dir = conf.img_dir
        self.OF_dir = OF_dir
        self.mask_dir = conf.mask_dir
        self.categories = os.listdir(self.OF_dir)
        trainii = ['scooter-gray','soccerball','stroller','surf','swing','tennis','train']
        valii = ['paragliding-launch','parkour','scooter-black','soapbox']
        self.categories = ['soapbox']
        # self.categories = ['elephant']
        # self.categories = ['stroller']
        
        self.num_categories = len(self.categories)
        self.imgs_by_cat, self.OFs_by_cat, self.masks_by_cat = {}, {}, {}
        for cat in self.categories:
            self.imgs_by_cat[cat] = sorted(os.listdir(os.path.join(self.img_dir, cat)))
            self.OFs_by_cat[cat] = sorted(os.listdir(os.path.join(self.OF_dir, cat)))
            self.masks_by_cat[cat] = sorted(os.listdir(os.path.join(self.mask_dir, cat)))

        self.resolution = (conf.resolution)
        self.T = T
        self._opt = conf
        self._noof = False
        if 'noof' in self._opt.name:
            self._noof = True
        self.create_transform()
        
    

    def create_transform(self):
        transforms_list_img = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],\
                                 std=[0.5, 0.5, 0.5])
        ]
        
        transforms_list_flow = [transforms.ToTensor(), ]
        transforms_list_gray = [transforms.ColorJitter(0.3,0,0,0.5), transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
        geom_transforms_fg = [transforms.ColorJitter(0.3,0.3,0.5,0.05),transforms.RandomAffine(8,(0,0.025),(0.75,1.25)),transforms.RandomHorizontalFlip(0.5)]
        
        self.transform_img = transforms.Compose(transforms_list_img)
        self.transform_flow = transforms.ToTensor()
        if self._noof:
            self.transform_gray = transforms.Compose(transforms_list_gray)
        self.geom_transforms = transforms.RandomApply([ 
            transforms.RandomAffine(8,(0.0,0.02),(0.75,1.25)), transforms.RandomHorizontalFlip(0.5)], p=0.45)
        self.fg_geom_transforms = transforms.Compose(geom_transforms_fg)

    def __getitem__(self, idx):
        
        cat = self.categories[idx]
        self._cat = cat
        
        imgs_paths = self.imgs_by_cat[cat][0:self.T]
        imgs_paths = [os.path.join(self.img_dir, cat, x) for x in imgs_paths]
        OFs_paths = self.OFs_by_cat[cat][0:self.T]
        OFs_paths = [os.path.join(self.OF_dir, cat, x) for x in OFs_paths]

        imgs = []
        OFs = []
        warped_imgs = []
        gray_imgs = []
        for i in range(self.T):
            img = cv2.imread(imgs_paths[i])
            if(i==0):
                mask = cv2.imread(os.path.join(self.mask_dir, cat, self.masks_by_cat[cat][0]))
                # cv2.imwrite('./gt_mask_orisize.png', mask)
                masked_img = cv2.bitwise_and(img, mask)
                masked_img_ori = masked_img.copy()

                mask_resized = resize_img(mask, self.resolution)
                # cv2.imwrite('./gt_mask_resized.png', np.array(mask_resized))
                masked_img = resize_img(masked_img, self.resolution)
                
                masked_img = self.transform_img(masked_img)

                mask_bg = cv2.bitwise_not(mask)
                masked_img_bg = cv2.bitwise_and(img, mask_bg)
                
                mask_uni = mask[:,:,0]
                new_img = img.copy()
                for i in range(img.shape[0]):
                    idxs = np.where(mask_uni[i,:]==255)[0]
                    if(len(idxs)>0):
                        num_idxs = len(idxs)
                        new_img[i,idxs[0:num_idxs//2],:] = img[i,idxs[0]-3,:]
                        new_img[i, idxs[num_idxs//2 +1 :], :] = img[i, idxs[-1]+3,:]

                masked_img_bg = resize_img(new_img, self.resolution)
                masked_img_bg = self.transform_img(masked_img_bg)

            elif(i<=self.T-1):
                flow = readFlow(OFs_paths[i])
                u = flow[:,:,0]
                v = flow[:,:,1]
                flow_u_remaped = remap_values(u, -20, 20, 0, 255)
                flow_v_remaped = remap_values(v, -20, 20, 0, 255)
                flow_u_remaped = resize_gray_img(flow_u_remaped, self.resolution)
                flow_v_remaped = resize_gray_img(flow_v_remaped, self.resolution)
                joint_flow_shape = (np.array(flow_u_remaped).shape[0], np.array(flow_u_remaped).shape[1], 2)
                x = np.zeros(joint_flow_shape, dtype=np.uint8)
                x[:,:,0] = flow_u_remaped
                x[:,:,1] = flow_v_remaped
                x = self.transform_flow(x)
                OFs.append(x)

            img = resize_img(img, self.resolution)
            if self._noof:
                gray_imgs.append(self.transform_gray(img))
            img = self.transform_img(img)
            imgs.append(img)
            

        sample = {}
        sample["imgs"] = imgs # range -1,1
        sample["OFs"] = OFs # range 0,1
        # sample["warped_imgs"] = warped_imgs # range -1,1
        sample["mask_f"] = masked_img # range -1,1
        sample["mask_b"] = masked_img_bg # range -1,1
        sample["mask"] = self.transform_flow(mask_resized) # Mask goes from 0 to 1 so we just apply ToTensor() transform
        
        sample["transformed_mask"] = self.transform_flow(self.geom_transforms(resize_img(mask, self.resolution)))
        sample["transformed_fg"] = self.transform_img(self.fg_geom_transforms(resize_img(masked_img_ori, self.resolution)))
        if self._noof:
            sample['gray_imgs'] = gray_imgs
        return sample

    def __len__(self):
        return self.num_categories



class ValDavisDataset(data.Dataset):
    
    def __init__(self, conf, T, OF_dir):
        super(ValDavisDataset, self).__init__()
        
        self.img_dir = conf.img_dir
        self.OF_dir = OF_dir
        self.mask_dir = conf.mask_dir
        self.categories = os.listdir(self.OF_dir)
        # only in training results
        # idx_of_dogagility = self.categories.index('dog-agility')
        # del self.categories[idx_of_dogagility]
        # self.categories = ['bmx-bumps']
        # self.categories = ['swing']
        self.num_categories = len(self.categories)
        self.imgs_by_cat, self.OFs_by_cat, self.masks_by_cat = {}, {}, {}
        for cat in self.categories:
            self.imgs_by_cat[cat] = sorted(os.listdir(os.path.join(self.img_dir, cat)))
            self.OFs_by_cat[cat] = sorted(os.listdir(os.path.join(self.OF_dir, cat)))
            self.masks_by_cat[cat] = sorted(os.listdir(os.path.join(self.mask_dir, cat)))

        
        self.resolution = (conf.resolution)
        self.T = T
        self.create_transform()
        self.my_cat = None
    def create_transform(self):
        transforms_list_img = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],\
                                 std=[0.5, 0.5, 0.5])
        ]
        transforms_list_flow = [transforms.ToTensor(), ]

        self.transform_img = transforms.Compose(transforms_list_img)
        self.transform_flow = transforms.ToTensor()

    def __getitem__(self, idx):
        # Be Careful if num_workers > 1, won't work
        cat = self.categories[idx]
        
        imgs_paths = self.imgs_by_cat[cat][0:self.T]
        imgs_paths = [os.path.join(self.img_dir, cat, x) for x in imgs_paths]
        OFs_paths = self.OFs_by_cat[cat][0:self.T]
        OFs_paths = [os.path.join(self.OF_dir, cat, x) for x in OFs_paths]
        # print("Category: ", cat, " num flows: ", len(OFs_paths))
        # print(OFs_paths[-1])
        imgs = []
        OFs = []
        gt_masks = []
        
        
        for i in range(self.T):
            img = cv2.imread(imgs_paths[i])
            if(i==0):
                mask = cv2.imread(os.path.join(self.mask_dir, cat, self.masks_by_cat[cat][0]))
                # cv2.imwrite('./gt_mask_orisize.png', mask)
                masked_img = cv2.bitwise_and(img, mask)
                
                masked_img_ori = masked_img.copy()
                mask_resized = resize_img(mask, self.resolution)
                # cv2.imwrite('./gt_mask_resized.png', mask_resized)
                masked_img = resize_img(masked_img, self.resolution)
                masked_img = self.transform_img(masked_img)

                mask_bg = cv2.bitwise_not(mask)
                masked_img_bg = cv2.bitwise_and(img, mask_bg)
                
                mask_uni = mask[:,:,0]
                new_img = img.copy()
                for a in range(img.shape[0]):
                    idxs = np.where(mask_uni[a,:]==255)[0]
                    if(len(idxs)>0):
                        num_idxs = len(idxs)
                        new_img[a,idxs[0:num_idxs//2],:] = img[a,idxs[0]-3,:]
                        new_img[a, idxs[num_idxs//2 +1 :], :] = img[a, idxs[-1]+3,:]

                masked_img_bg = resize_img(new_img, self.resolution)
                masked_img_bg = self.transform_img(masked_img_bg)

            elif(i<=self.T-1):
                flow = readFlow(OFs_paths[i])
            
                u = flow[:,:,0]
                v = flow[:,:,1]
                flow_u_remaped = remap_values(u, -20, 20, 0, 255)
                flow_v_remaped = remap_values(v, -20, 20, 0, 255)
                flow_u_remaped = resize_gray_img(flow_u_remaped, self.resolution)
                flow_v_remaped = resize_gray_img(flow_v_remaped, self.resolution)
                joint_flow_shape = (np.array(flow_u_remaped).shape[0], np.array(flow_u_remaped).shape[1], 2)
                x = np.zeros(joint_flow_shape, dtype=np.uint8)
                x[:,:,0] = flow_u_remaped
                x[:,:,1] = flow_v_remaped
                x = self.transform_flow(x)
                OFs.append(x)
            gt_masks.append(self.transform_flow(resize_img(cv2.imread(os.path.join(self.mask_dir, cat, self.masks_by_cat[cat][i])), self.resolution)))
            img = resize_img(img, self.resolution)
            img = self.transform_img(img)
            imgs.append(img)

        sample = {}
        sample["imgs"] = imgs # range -1,1
        sample["OFs"] = OFs # range 0,1
        sample["mask_f"] = masked_img # range -1,1
        sample["mask_b"] = masked_img_bg # range -1,1
        sample["mask"] = self.transform_flow(mask_resized) # Mask goes from 0 to 1 so we just apply ToTensor() transform
        sample["gt_masks"] = gt_masks
        sample["video_name"] = cat
        return sample

    def __len__(self):
        return self.num_categories



def extract_bg_patches(first_fg, first_bg, batch_size = 5):
    
    kh, kw, stride_h, stride_w = 60,112,20,36
    kernel = torch.ones(1,3,60,112)
    output = F.conv2d(first_fg + 1.0, kernel, stride=(20,36))
    convsize = output.size()[-1]
    indexes = torch.le(output, 0.001)
    
    N = 10
    nonzero_indexes = []
    nonzero_elements = [0] * batch_size
    for i in range(batch_size):
        cerveseta = indexes[i,0,:,:]
        nonz = torch.nonzero(cerveseta) # [nelem,2]
        nonzero_indexes.append(nonz)
        nelem = nonz.size()[0]
        nonzero_elements[i] = nelem
        N = min(nelem, N)

    image_patches = torch.zeros(batch_size,N,3,kh,kw)
    to_image_coords = torch.tensor([stride_h, stride_w]).expand((N,2))
    img_indexes = torch.zeros(batch_size, N, 2)

    for i in range(batch_size):
        random_integers = np.unique(np.random.randint(0,nonzero_elements[i],N))
        while(random_integers.shape[0]<N):
            random_integers = np.unique(np.random.randint(0,nonzero_elements[i],N))
        conv_indexes = nonzero_indexes[i][random_integers]
        img_indexes[i, :, :] = conv_indexes * to_image_coords

    for b in range(batch_size):
        for n in range(N):
            P1 = int(img_indexes[b,n,0])
            P2 = int(img_indexes[b,n,1])
            image_patches[b, n, :, :, :] = first_bg[b, :, P1 : P1 + kh, P2:P2+kw]

    return image_patches

    # sample_patch = cv2.cvtColor(tensor2im(image_patches[0,2,:,:,:]),cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./imgs/sample_patch.jpg', sample_patch)
    # sample_patch2 = cv2.cvtColor(tensor2im(image_patches[1,2,:,:,:]),cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./imgs/sample_patch2.jpg', sample_patch2)
    # sample_patch3 = cv2.cvtColor(tensor2im(image_patches[2,2,:,:,:]),cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./imgs/sample_patch3.jpg', sample_patch3)
    # sample_patch4 = cv2.cvtColor(tensor2im(image_patches[3,2,:,:,:]),cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./imgs/sample_patch4.jpg', sample_patch4)
    # sample_patch5 = cv2.cvtColor(tensor2im(image_patches[4,2,:,:,:]),cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./imgs/sample_patch5.jpg', sample_patch5)

    
if __name__ == '__main__':
    d = DavisDataset('./VFG/options/configs.json', T=3,OF_dir='/data/Ponc/DAVIS/OpticalFlows/')
    dl = data.DataLoader(d,batch_size=5)
    bat = next(iter(dl))
    # print(bat.keys())
    T = d.T
    import torch
    import torch.nn.functional as F
    extract_bg_patches(bat['mask_f'], bat['imgs'][0])
    print(bat['mask_f'].size())
    print(torch.min(bat['mask_f']))

    # for t in range(T):
        # print("Fmask")
        # print(bat['mask_f'].size())
        # print(torch.min(bat['mask_f']))

    
    # print(bat['imgs'][0][0,:,0,0])
    # print(bat['OFs'][0][0,:,0,0])
    # print(bat['warped_imgs'][0][0,:,0,0])
    # print(bat['mask'][0][:,0,0])
    # tensor([-0.6000, -0.6392, -0.6627])
    # tensor([0.5569, 0.5216])
    # tensor([-0.9373, -0.9451, -0.9451])
    # tensor([-1., -1., -1.])
    # for i, val in enumerate(dl):
        # print(i, val.keys())
        # print(val['imgs'])
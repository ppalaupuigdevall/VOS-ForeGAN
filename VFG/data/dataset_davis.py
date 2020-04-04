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

def resize_img(img, size):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    # img_cv2_resized = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2BGR)
    return img_pil_resized

def resize_gray_img(img, size):
    img_pil = Image.fromarray(img)
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    # img_cv2_resized = np.array(img_pil_resized)
    return img_pil_resized

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
    
    def __init__(self, conf):
        super(DavisDataset, self).__init__()
        
        with open(conf, 'r') as f:
            config = json.load(f)["dataset"]
        self.img_dir = config['img_dir']  
        self.OF_dir = config['OF_dir']
        self.mask_dir = config['mask_dir']
        self.categories = os.listdir(self.OF_dir)
        self.num_categories = len(self.categories)
        print(self.num_categories)
        self.imgs_by_cat, self.OFs_by_cat, self.masks_by_cat = {}, {}, {}
        for cat in self.categories:
            self.imgs_by_cat[cat] = sorted(os.listdir(os.path.join(self.img_dir, cat)))
            self.OFs_by_cat[cat] = sorted(os.listdir(os.path.join(self.OF_dir, cat)))
            self.masks_by_cat[cat] = sorted(os.listdir(os.path.join(self.mask_dir, cat)))

        self.resolution = (config['resolution'][0], config['resolution'][1])
        self.T = config["T"]
        self.create_transform()
        self.shift = False #TODO: set variable to give mask in sample or not depending on score of Discriminator 
    

    def create_transform(self):
        """NOTE: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]"""
        transforms_list_img = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],\
                                 std=[0.5, 0.5, 0.5])
        ]
        transforms_list_flow = [transforms.ToTensor(), ]

        self.transform_img = transforms.Compose(transforms_list_img)
        self.transform_flow = transforms.ToTensor()

    def __getitem__(self, idx):
        cat = self.categories[idx]
        imgs_paths = self.imgs_by_cat[cat][0:self.T]
        imgs_paths = [os.path.join(self.img_dir, cat, x) for x in imgs_paths]
        OFs_paths = self.OFs_by_cat[cat][0:self.T]
        OFs_paths = [os.path.join(self.OF_dir, cat, x) for x in OFs_paths]

        imgs = []
        OFs = []
        warped_imgs = []
        for i in range(self.T):
            img = cv2.imread(imgs_paths[i])
            if(i==0):
                mask = cv2.imread(os.path.join(self.mask_dir, cat, self.masks_by_cat[cat][0]))
                masked_img = cv2.bitwise_and(img, mask)
                masked_img_ori = masked_img.copy()
                
                masked_img = resize_img(masked_img, self.resolution)
                masked_img = self.transform_img(masked_img)

                mask_bg = cv2.bitwise_not(mask)
                masked_img_bg = cv2.bitwise_and(img, mask_bg)
                masked_img_bg = resize_img(masked_img_bg, self.resolution)
                masked_img_bg = self.transform_img(masked_img_bg)


            elif(i<self.T-1):
                flow = readFlow(OFs_paths[i])
            
                warped_img = warp_flow(masked_img_ori, flow)
                masked_img_ori = warped_img.copy()
                warped_img = resize_img(warped_img, self.resolution)
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
                warped_img = self.transform_img(warped_img)
                warped_imgs.append(warped_img)
                

            img = resize_img(img, self.resolution)
            img = self.transform_img(img)
            imgs.append(img)

        sample = {}
        sample["imgs"] = imgs
        sample["OFs"] = OFs
        sample["warped_imgs"] = warped_imgs
        sample["mask_f"] = masked_img
        sample["mask_b"] = masked_img_bg
        
        return sample

    def __len__(self):
        return self.num_categories



def extract_bg_patches(first_fg):
    
    kernel = torch.ones(1,3,60,112)
    output = F.conv2d(first_fg + 1.0, kernel, stride=(20,36))
    print(output.size())
    print(output[0,:,:,:])
    # TODO: We have 

if __name__ == '__main__':
    d = DavisDataset('./VFG/options/configs.json')
    dl = data.DataLoader(d, batch_size=5)
    bat = next(iter(dl))
    print(bat.keys())
    T = d.T
    import torch
    import torch.nn.functional as F
    extract_bg_patches(bat['mask_f'])
    for t in range(T):
        print("Fmask")
        print(bat['mask_f'].size())
        print(torch.min(bat['mask_f']))

    
    # print(bat['imgs'][0][0,:,0,0])
    # print(bat['OFs'][0][0,:,0,0])
    # print(bat['warped_imgs'][0][0,:,0,0])
    # print(bat['mask'][0][:,0,0])
    # tensor([-0.6000, -0.6392, -0.6627])
    # tensor([0.5569, 0.5216])
    # tensor([-0.9373, -0.9451, -0.9451])
    # tensor([-1., -1., -1.])
    for i, val in enumerate(dl):
        print(i, val.keys())
        # print(val['imgs'])
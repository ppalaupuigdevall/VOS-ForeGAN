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
        self.categories = os.listdir(self.OF_dir)
        self.num_categories = len(self.categories)
        self.imgs_by_cat, self.OFs_by_cat = {}, {}
        for cat in self.categories:
            self.imgs_by_cat[cat] = sorted(os.listdir(os.path.join(self.img_dir, cat)))
            self.OFs_by_cat[cat] = sorted(os.listdir(os.path.join(self.OF_dir, cat)))

        self.resolution = (config['resolution'][0], config['resolution'][1])
        self.T = config["T"]
        self.create_transform() 
    

    def create_transform(self):
        transforms_list_img = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],\
                                 std=[0.5, 0.5, 0.5])
        ]
        transforms_list_flow = [transforms.ToTensor()]

        self.transform_img = transforms.Compose(transforms_list_img)
        self.transform_flow = transforms.Compose(transforms_list_flow)

    def __getitem__(self, idx):
        cat = self.categories[idx]
        imgs_paths = self.imgs_by_cat[cat][0:self.T]
        OFs_paths = self.OFs_by_cat[cat][0:self.T]
        imgs = []
        OFs = []
        warped_imgs = []
        for i in range(self.T):
            img = cv2.imread(imgs_paths[i])
            if(i<self.T-1):
                flow = readFlow(OFs_paths[i])
                warped_img = resize_img(warp_flow(img, flow))
                u = flow[:,:,0]
                v = flow[:,:,1]
                flow_u_remaped = remap_values(u, -20, 20, 0, 255)
                flow_v_remaped = remap_values(v, -20, 20, 0, 255)
                flow_u_remaped = resize_gray_img(flow_u_remaped, self.resolution)
                flow_v_remaped = resize_gray_img(flow_v_remaped, self.resolution)
                joint_flow_shape = (flow_u_remaped.shape[0], flow_u_remaped.shape[1], 2)
                x = np.zeros(joint_flow_shape)
                x[:,:,0] = flow_u_remaped
                x[:,:,1] = flow_v_remaped
                
                OFs.append(x)

                warped_imgs.append(warped_img)
            img = resize_img(img, self.resolution)

            imgs.append(img)


        


    def __len__(self):
        return self.num_categories



if __name__ == '__main__':
    d = DavisDataset('./VFG/options/configs.json')
    
import torch
import torch.nn.functional as F
from VFG.data.dataset_davis import *

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


batch_size = 4
d = DavisDataset('./VFG/options/configs.json')
dl = data.DataLoader(d, batch_size=batch_size)
bat = next(iter(dl))
x = bat['imgs'][0]



def extract_img_patches(x, ch=3, height=224, width=416, kh=60, kw=112,dh=20, dw=36):
    patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.view(-1, ch, kh, kw)
    return patches

channels = 3
height, width = 224,416 

kh, kw = 60,112 
dh, dw = 5, 10
dh, dw = 30, 54 # this should be a more feasible stride

# Pad tensor to get the same output
# x = F.pad(x, (1, 1, 1, 1))

# get all image windows of size (kh, kw) and stride (dh, dw)
patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
patch = patches[1,:,2,3,:,:]
patch_img = tensor2im(patch)
cv2.imwrite('./imgs/img_patch.jpg', patch_img)
print(patches.shape)  # [128, 16, 32, 32, 3, 3]
# Permute so that channels are next to patch dimension
patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [128, 32, 32, 16, 3, 3]
# View as [batch_size, height, width, channels*kh*kw]
# patches = patches.view(*patches.size()[:3], -1)
# print(patches.shape)
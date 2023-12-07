#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import random
from model import DDPM 
import model

import torch

n_T = 500 # 500
device=torch.device("cpu")
n_classes = 20
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = True



ddpm = DDPM(nn_model=model.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=torch.device("cpu"), drop_prob=0.1)

ddpm.load_state_dict(torch.load("new_model_9.pth", map_location='cpu'))




target_digit = random.randint(0, 9)


is_flipped = random.randint(0, 1)  

if is_flipped:
    context_label = target_digit + 10
else: 
    context_label = target_digit

print(context_label)

tensor = torch.tensor([context_label], dtype=torch.long)
print(tensor.shape)
print(tensor)
ddpm.eval()


with torch.no_grad():
    x_gen, x_gen_store = ddpm.sample_single( 1, (1,28,28), device, tensor, guide_w = 4.0)
    #numpy_array = x_gen[0].cpu().detach().numpy()
    numpy_array = x_gen[0].squeeze(0).cpu().detach().numpy()
    # Display the image using plt.imshow
    plt.imshow(numpy_array, cmap='gray')
    plt.axis('off')

    plt.show()


# In[ ]:





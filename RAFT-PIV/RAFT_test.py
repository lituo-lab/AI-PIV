import torch
import numpy as np
import argparse
from data import MyDataset
from flowNetsRAFT import RAFT
import matplotlib.pyplot as plt


device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--amp', type=eval, default=False, help='Wether to use auto mixed precision')
parser.add_argument('--iters', default=12, type=int, help='number of update steps in ConvGRU')
args = parser.parse_args()

model = RAFT().to(device)
model.load_state_dict(torch.load('ckpt.tar')['model_state_dict'],strict=False)

data = MyDataset('.\DNS_turbulence')
images, flows = data[1]
images = images.unsqueeze(0).to(device)
flows = flows.unsqueeze(0).to(device)


with torch.no_grad():
    pred_flows = model(images, flows*0, args=args)[0][-1]

u_pre = pred_flows[0,0]
v_pre = pred_flows[0,1]
u_act = flows[0,0]
v_act = flows[0,1]


plt.figure(figsize=(12.5,10))

plt.subplot(2, 2, 1)
plt.contourf(u_act,levels=np.linspace(u_act.min(),u_act.max(),100),cmap='jet')
plt.colorbar()
plt.title('u_act')

plt.subplot(2, 2, 2)
plt.contourf(u_pre,levels=np.linspace(u_act.min(),u_act.max(),100),cmap='jet')
plt.colorbar()
plt.title('u_pre')

plt.subplot(2, 2, 3)
plt.contourf(v_act,levels=np.linspace(v_act.min(),v_act.max(),100),cmap='jet')
plt.colorbar()
plt.title('v_act')

plt.subplot(2, 2, 4)
plt.contourf(v_pre,levels=np.linspace(v_act.min(),v_act.max(),100),cmap='jet')
plt.colorbar()
plt.title('v_pre')

plt.show()










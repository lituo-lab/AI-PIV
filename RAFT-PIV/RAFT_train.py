import torch
import numpy as np
from flowNetsRAFT import RAFT
from tqdm import tqdm
from data import MyDataset
from torch import nn
import argparse
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
scaler = GradScaler()
autocast = torch.cuda.amp.autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--amp', type=eval, default=False, help='Wether to use auto mixed precision')
parser.add_argument('--iters', default=12, type=int, help='number of update steps in ConvGRU')
parser.add_argument('--init_lr', default=0.0001, type=float, help='initial learning rate')
args = parser.parse_args()


root = '.\DNS_turbulence'
data_loader = DataLoader(MyDataset(root), batch_size=1, shuffle=True)

model = RAFT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
model.load_state_dict(torch.load('ckpt.tar')['model_state_dict'],strict=False)

num_epochs = 100

for epoch in range(num_epochs):

    model.train()
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    epoch_loss = []
    for i, (images, flows) in loop:
        images, flows = images.to(device), flows.to(device)
        with autocast(enabled=args.amp):
            pred_flows = model(images, flows, args=args)

            training_loss, metrics = pred_flows[1]
            
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            # scaler.scale(training_loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # scaler.step(optimizer)
            # scaler.update() 
            
            loop.set_postfix_str(
                'loss: ' + "{:10.6f}".format(metrics['epe']))
















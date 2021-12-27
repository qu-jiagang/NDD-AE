import torch
import torch.nn as nn
import numpy as np


class DMDAE(nn.Module):
    def forward(self, x):
        channels = x.size(0)
        r1 = self.fcl1(self.encoder_layer(x).view(channels,-1))
        x_reconst_1 = self.decoder_layer(self.fcl3(r1).view(channels,512,3,6))
        return x_reconst_1


net = torch.load('NDM-AE.net')
net.eval()

Nx,Ny = 192,384
data = np.fromfile('../../dataset/periodic.dat').reshape([1000,1,Nx,Ny])

recon = np.zeros([1000,1,Nx,Ny])
recon[0] = data[0]
for i in range(1000-1):
    recon_t = net(torch.from_numpy(recon[i:i+1]).float().cuda())
    recon[i+1:i+2] = recon_t.cpu().data.numpy()
    print(i)
    del recon_t

print(recon.shape)
recon.tofile('NDM-AE-AR-reonst.dat') # (1000, 1, 192, 384)

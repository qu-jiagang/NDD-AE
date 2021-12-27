import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class DMDAE(nn.Module):
    def forward(self, x):
        channels = x.size(0)
        out = self.encoder_layer(x)
        z1 = self.fcl1(out.view(channels,-1))
        z2 = self.fcl2(z1)
        out2_1 = self.fcl3(z2[:,0:1]).view(channels,512,3,6)
        out2_2 = self.fcl4(z2[:,1:2]).view(channels,512,3,6)
        x2_1 = self.decoder_layer(out2_1)
        x2_2 = self.decoder_layer(out2_2)
        x_reconst_2 = x2_1 + x2_2

        return x_reconst_2


net = torch.load('NDD-AE.net')
net.eval()

Nx,Ny = 192,384
data = np.fromfile('../../dataset/periodic.dat').reshape([1000, 1, Nx, Ny])

recon = np.zeros([1000,1,Nx,Ny])
recon[0] = data[0]
for i in range(1000-1):
    recon_t = net(torch.from_numpy(recon[i:i+1]).float().cuda())
    recon[i+1:i+2] = recon_t.cpu().data.numpy()
    del recon_t
    print(i)

# save data
recon.tofile('NDD-AE-AR-reconst.dat') # [1000,1,Nx,Ny]

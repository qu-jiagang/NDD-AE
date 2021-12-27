import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class DMDAE(nn.Module):
    def forward(self, x):
        channels = x.size(0)
        out = self.encoder_layer(x)
        z1 = self.fcl1(out.view(channels,-1))
        out1_1 = self.fcl3(z1[:,0:1]).view(channels,512,3,6)
        out1_2 = self.fcl4(z1[:,1:2]).view(channels,512,3,6)
        x1_1 = self.decoder_layer(out1_1)
        x1_2 = self.decoder_layer(out1_2)
        x_reconst_1 = x1_1 + x1_2
        return x_reconst_1, z1, x1_1, x1_2


net = torch.load('NDD-AE.net')
net.eval()

Nx,Ny = 192,384
data = np.fromfile('../../dataset/periodic.dat').reshape([1000, 1, Nx, Ny])

recon = np.zeros([1000,1,Nx,Ny])
z = np.zeros([1000,2])
x = np.zeros([1000,2,Nx,Ny])
for i in range(1000):
    recon_t, z_t, x1_t, x2_t = net(torch.from_numpy(data[i:i+1]).float().cuda())
    recon[i:i+1] = recon_t.cpu().data.numpy()
    z[i:i+1] = z_t.cpu().data.numpy()
    x[i:i+1,0:1] = x1_t.cpu().data.numpy()
    x[i:i+1,1:2] = x2_t.cpu().data.numpy()
    del recon_t, z_t, x1_t, x2_t
    print(i)

# save data
recon.tofile('NDD-AE-MD-reconst.dat') # [1000,1,Nx,Ny]
z.tofile('NDD-AE-MD-eigs.dat') # [1000,2]
x.tofile('NDD-AE-MD-modes.dat') # [1000,2,Nx,Ny]

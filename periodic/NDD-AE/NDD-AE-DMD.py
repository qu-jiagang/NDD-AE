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
z_t = net.fcl1(net.encoder_layer(torch.from_numpy(data[0:1]).float().cuda()).view(1,-1))
z[0:1] = z_t.cpu().data.numpy()
recon[0] = data[0]
for i in range(1000-1):
    z_t = net.fcl2(z_t)
    x1 = net.decoder_layer(net.fcl3(z_t[:,0:1]).view(1,512,3,6))
    x2 = net.decoder_layer(net.fcl4(z_t[:,1:2]).view(1,512,3,6))
    recon_t = x1 + x2

    recon[i+1:i+2] = recon_t.cpu().data.numpy()
    z[i+1:i+2] = z_t.cpu().data.numpy()

    del recon_t, x1, x2
    print(i)

# save data
recon.tofile('NDD-AE-DMD-reconst.dat') # [1000,1,Nx,Ny]
z.tofile('NDD-AE-DMD-eigs.dat') # [1000,2]

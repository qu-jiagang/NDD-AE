import torch
import torch.nn as nn
import numpy as np


class DMDAE(nn.Module):
    def forward(self, x):
        channels = x.size(0)
        r1 = self.fcl1(self.encoder_layer(x).view(channels,-1))
        x1_1 = self.decoder_layer(self.fcl3(r1[:,0:1]).view(channels,512,3,6))
        x1_2 = self.decoder_layer(self.fcl4(r1[:,1:2]).view(channels,512,3,6))
        x1_3 = self.decoder_layer(self.fcl5(r1[:,2:3]).view(channels,512,3,6))
        x_reconst_1 = x1_1 + x1_2 + x1_3
        return x_reconst_1, x1_1, x1_2, x1_3, r1


net = torch.load('NDD-AE.net')
net.eval()

Nx,Ny = 384,768
data = np.fromfile('../../dataset/transient.dat').reshape([1500,1,Nx,Ny])[300:1200]
input = data[0].reshape([1,1,Nx,Ny])


recon = np.zeros([900,1,Nx,Ny])
z = np.zeros([900,3])
x = np.zeros([900,3,Nx,Ny])

for i in range(900):
    recon_t, x1_t, x2_t, x3_t, z_t = net(torch.from_numpy(data[i:i+1]).float().cuda())

    x[i:i+1, 0:1] = x1_t.cpu().data.numpy()
    x[i:i+1, 1:2] = x2_t.cpu().data.numpy()
    x[i:i+1, 2:3] = x3_t.cpu().data.numpy()
    recon[i:i+1] = recon_t.cpu().data.numpy()
    z[i:i+1] = z_t.cpu().data.numpy()

    print(i)
    del recon_t, x1_t, x2_t, x3_t, z_t


recon.tofile('NDD-AE-MD-reonst.dat')
z.tofile('NDD-AE-eigs.dat')
x.tofile('NDD-AE-modes.dat')

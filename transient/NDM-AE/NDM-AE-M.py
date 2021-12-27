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

Nx,Ny = 384,768
data = np.fromfile('../../dataset/transient.dat').reshape([1500,1,Nx,Ny])[300:1200]

recon = np.zeros([900,1,Nx,Ny])
for i in range(900):
    recon_t = net(torch.from_numpy(data[i:i+1]).float().cuda())
    recon[i:i+1] = recon_t.cpu().data.numpy()
    print(i)
    del recon_t

print(recon.shape)
recon.tofile('NDM-AE-M-reonst.dat') # (900, 1, 384, 768)

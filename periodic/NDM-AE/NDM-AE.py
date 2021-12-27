import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DMDAE(nn.Module):
    def __init__(self):
        super(DMDAE, self).__init__()

        self.encoder_layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.fcl1 = nn.Sequential(
            nn.Linear(512*3*6, 2),
        )

        self.fcl2 = nn.Sequential(
            nn.Linear(2,128),
            nn.Linear(128,2),
        )

        self.fcl3 = nn.Sequential(
            nn.Linear(2, 1000),
            nn.GELU(),
            nn.Linear(1000, 4096),
            nn.GELU(),
            nn.Linear(4096, 512*3*6),
        )

        self.decoder_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1, padding_mode='replicate'),
        )

    def forward(self, x, x_shift):
        channels = x.size(0)
        out = self.encoder_layer(x)
        z1 = self.fcl1(out.view(channels,-1))
        z2 = self.fcl2(z1)
        out1 = self.fcl3(z1).view(channels,512,3,6)
        out2 = self.fcl3(z2).view(channels,512,3,6)
        x_reconst_1 = self.decoder_layer(out1)
        x_reconst_2 = self.decoder_layer(out2)

        z2_from_shift = self.fcl1(self.encoder_layer(x_shift).view(channels,-1))

        return x_reconst_1, x_reconst_2, z2, z2_from_shift


def loss_func(recon_x_1, x_1, recon_x_2, x_2, z2, z2_from_shift):
    loss1 = F.mse_loss(recon_x_1, x_1)
    loss2 = F.mse_loss(recon_x_2, x_2)
    loss3 = F.mse_loss(z2_from_shift, z2)
    return loss1+loss2+loss3, loss1+loss2, loss3


Nx,Ny = 192,384
net = DMDAE().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

data = np.fromfile('../../dataset/periodic.dat').reshape([1000, 1, Nx, Ny])
data_x = data[:-1]
data_x_shift = data[1:]
data_x = data_x.reshape([-1,1,Nx,Ny])
data_x_shift = data_x_shift.reshape([-1,1,Nx,Ny])

data_x = torch.from_numpy(data_x).float()
data_x_shift = torch.from_numpy(data_x_shift).float()

database = TensorDataset(data_x,data_x_shift)
train_loader = DataLoader(
    dataset=database,
    batch_size=50,
    shuffle=True,
    drop_last=True
)

_loss, loss_base, epoch = 1.0, 0.1, -1
while epoch < 2000:
    _BCE, _KLD = 0, 0
    _loss,_loss1,_loss2 = 0,0,0
    epoch += 1
    for i, data_cut in enumerate(train_loader):
        input_data = Variable(data_cut[0]).cuda()
        output_data = Variable(data_cut[1]).cuda()

        optimizer.zero_grad()
        x_reconst_1, x_reconst_2, z2, z2_from_shift = net(input_data, output_data)
        loss, loss1, loss2 = loss_func(x_reconst_1, input_data, x_reconst_2, output_data, z2, z2_from_shift)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1)
        optimizer.step()

        _loss += loss.cpu().detach().numpy()
        _loss1 += loss1.cpu().detach().numpy()
        _loss2 += loss2.cpu().detach().numpy()

    scheduler.step()

    _loss /= i + 1
    _loss1 /= i + 1
    _loss2 /= i + 1

    if epoch % 1 == 0:
        print(epoch,_loss, _loss1, _loss2)

    if _loss < loss_base:
        loss_base = _loss
        torch.save(net, 'NDM-AE.net')

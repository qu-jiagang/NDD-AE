import numpy as np
import matplotlib.pyplot as plt

def error(x,y):
    error = np.mean((x-y)**2)
    return error

Nx,Ny = 192,384

data = np.fromfile('../dataset/periodic.dat').reshape([1000, Nx, Ny])
NDD_AE_MD_reconst = np.fromfile('NDD-AE/NDD-AE-MD-reconst.dat').reshape([1000, Nx, Ny])
POD_reconst = np.fromfile('POD/POD_reconst.dat').reshape([1000, Nx, Ny])
DMD_reconst = np.fromfile('DMD/DMD_reonst.dat').reshape([Nx, Ny, 1000]).transpose([2, 0, 1])


plt.imshow(DMD_reconst[0],vmin=-3,vmax=3,cmap=plt.cm.RdBu_r)
plt.show()

error_MD = error(data,NDD_AE_MD_reconst)
error_POD = error(data,POD_reconst)
error_DMD = error(data,DMD_reconst)

print(error_MD)
print(error_POD)
print(error_DMD)
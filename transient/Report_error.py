import numpy as np
import matplotlib.pyplot as plt


def error(x,y):
    error = np.mean((x-y)**2)
    return error


Nx,Ny = 384, 768
data = np.fromfile('../dataset/transient.dat').reshape([1500, Nx, Ny])[300:1200]
NDM_AE_M_reconst = np.fromfile('./NDM-AE/NDM-AE-M-reonst.dat').reshape([900, Nx, Ny])
NDD_AE_MD_reconst = np.fromfile('./NDD-AE/NDD-AE-MD-reonst.dat').reshape([900, Nx, Ny])
POD_reconst = np.fromfile('./POD/POD_reconst.dat').reshape([900, Nx, Ny])
DMD_reconst = np.fromfile('./DMD/DMD_reonst.dat').reshape([Nx, Ny, 900]).transpose([2,0,1])


error_NDM = error(data,NDM_AE_M_reconst)
error_NDD = error(data,NDD_AE_MD_reconst)
error_POD = error(data,POD_reconst)
error_DMD = error(data,DMD_reconst)

print(error_POD)
print(error_DMD)
print(error_NDD)
print(error_NDM)

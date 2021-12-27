import numpy as np
from numpy import dot, multiply, power
from numpy.linalg import inv, pinv
import dask.array as da


Nx, Ny = 384, 768
data = np.fromfile('../../dataset/transient.dat').reshape([1500,Nx,Ny])[300:1200]

X = data[:-1].reshape([-1,Nx*Ny]).transpose([1,0])        # data for before linear operation
X_prime = data[1:].reshape([-1,Nx*Ny]).transpose([1,0])   # data for after linear operation

X = da.from_array(X)

r = 11  # selected rank
U,S,Vh = da.linalg.svd(X)
U,S,Vh = np.array(U), np.array(S), np.array(Vh)
Ur = U[:,:r]
Sr = np.diag(S)[:r,:r]
Vr = Vh.conj().T[:,:r]

A_tilde = dot(dot(dot(Ur.conj().T, X_prime), Vr), inv(Sr))
mu, W = np.linalg.eig(A_tilde)

idx = mu.argsort()[::-1]
mu = mu[idx]
W = W[:,idx]

Phi = dot(dot(dot(X_prime, Vr), inv(Sr)), W)

b = dot(pinv(Phi), X[:,0])
Psi = np.zeros([r, 900], dtype='complex')
t = np.linspace(30,120,900,endpoint=False)
dt = t[1]-t[0]
for i,_t in enumerate(t):
    Psi[:,i] = multiply(power(mu, _t/dt), b)

Reconst = np.real(dot(Phi,Psi)).reshape([Nx,Ny,900])

# save data
print(Phi.shape) # (294912, 11) = (384, 768, 11)
print(Psi.shape) # (11, 900)
print(Reconst.shape) # (384, 768, 900)
Phi.tofile('DMD_eigs.dat')
Psi.tofile('DMD_modes.dat')
Reconst.tofile('DMD_reonst.dat')

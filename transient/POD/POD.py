from sklearn.decomposition import PCA
import numpy as np


Nx, Ny = 384,768
periodic = np.fromfile('../../dataset/transient.dat').reshape([1500,Nx,Ny])[300:1200]

# obtain POD with PCA
X = periodic.reshape([-1,Nx*Ny])
pca = PCA(n_components=10,svd_solver='full')
pca.fit(X)

# POD eigenvalues
POD_eig = pca.transform(X)

# POD modes
POD_modes = pca.components_
POD_modes = POD_modes.reshape([10,Nx,Ny])

# POD reconstruction
POD_reconst = pca.inverse_transform(POD_eig)
POD_reconst = POD_reconst.reshape([-1,Nx,Ny])

# Save data
print(POD_eig.shape) # (900, 10)
print(POD_modes.shape) # (10, 384, 768)
print(POD_reconst.shape) # (900, 384, 768)
POD_eig.tofile('POD_eig.dat')
POD_modes.tofile('POD_modes.dat')
POD_reconst.tofile('POD_reconst.dat')

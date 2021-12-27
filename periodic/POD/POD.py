from sklearn.decomposition import PCA
import numpy as np


Nx, Ny = 192,384
periodic = np.fromfile('../../dataset/periodic.dat').reshape([1000,Nx,Ny])

# obtain POD with PCA
X = periodic.reshape([-1,Nx*Ny])
pca = PCA(n_components=4,svd_solver='full')
pca.fit(X)

# POD eigenvalues
POD_eig = pca.transform(X)

# POD modes
POD_modes = pca.components_
POD_modes = POD_modes.reshape([4,Nx,Ny])

# POD reconstruction
POD_reconst = pca.inverse_transform(POD_eig)
POD_reconst = POD_reconst.reshape([-1,Nx,Ny])

# Save data
print(POD_eig.shape) # (1000, 4)
print(POD_modes.shape) # (4, 192, 384)
print(POD_reconst.shape) # (1000, 192, 384)
POD_eig.tofile('POD_eig.dat')
POD_modes.tofile('POD_modes.dat')
POD_reconst.tofile('POD_reconst.dat')

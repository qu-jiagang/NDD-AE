# Periodic Cylinder wake

## 1. POD

We obtain the POD results by performing PCA.

* retained number of POD modes: 4
* POD eigenvalues: `POD_eig.dat`
  * size: $1000\times 4$ 
* POD modes: `POD_modes.dat`
  * size: $4\times 192\times 384$ 
* POD reconstruction: `POD_reconst.dat`
  * size: $1000\times 192\times 384$ 

## 2. DMD

We perform DMD with SVD, and perform SVD with `dask`

* retained number of POD modes: 4
* POD eigenvalues: `DMD_eig.dat`
  * size: $1000\times 5$ 
* POD modes: `DMD_modes.dat`
  * size: $5\times 192\times 384$ 
* POD reconstruction: `DMD_reconst.dat`
  * size: $1000\times 192\times 384$ 

## 3. NDM-AE

## 4. NDD-AE


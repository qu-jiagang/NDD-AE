# Periodic Cylinder wake

## 1. POD

We obtain the POD results by performing PCA.

* retained number of POD modes: 4
* POD eigenvalues: `POD_eig.dat`
  * size: $1000\times 4$ 
* POD modes: `POD_modes.dat`
  * size: $4\times 192\times 384$ 
* POD reconstruction: `POD_reconst.dat`(Too large and Not shown)
  * size: $1000\times 192\times 384$ 

## 2. DMD

We perform DMD with SVD, and perform SVD with `dask`

* retained number of POD modes: 4
* POD eigenvalues: `DMD_eig.dat`
  * size: $1000\times 5$ 
* POD modes: `DMD_modes.dat`
  * size: $5\times 192\times 384$ 
* POD reconstruction: `DMD_reconst.dat` (Too large and Not shown)
  * size: $1000\times 192\times 384$ 

## 3. NDM-AE

nonlinear dynamic modelling autoencoder

Network: `NDM-AE.py`

`NDM-AE-M.py`: Return the latent representations from DNS data (snapshots)

`NDM-AE-AR.py`:

* predict the future state (snapshot at $t_{k+1}$) in AR manner (from snapshot at $t_k$)
* return the latent representations from the predictions (not DNS data)

## 4. NDD-AE

nonlinear dynamic decomposition autoencoder 

Network: `NDD-AE.py`

`NDD-AE-MD.py`: Return latent representations ($\boldsymbol r_{k}$)  and decomposed fields from DNS data ($\boldsymbol u_{k}$) 

`NDD-AE-AR.py`:

* predict the future state (snapshot at $t_{k+1}$) in AR manner (from snapshot at $t_k$)
* return the latent representations ($\boldsymbol r_{k+1}$) and decomposed fields  ($\hat{\boldsymbol u}_{k+1}$)  from the predictions  ($\boldsymbol u_{k+1}$)  (not DNS data)

`NDD-AE-DMD.py`: 

* predict the future state (snapshot at $t_{k+1}$) in DMD manner (from the latent representations at $t_k$)
* return the latent representations ($\boldsymbol r_{k+1}$) from the representations of previous state ($\boldsymbol r_{k}$)
* return the decomposed fields ($\hat{\boldsymbol u}_{k+1}$) from predicted representations ($\boldsymbol r_{k+1}$) 


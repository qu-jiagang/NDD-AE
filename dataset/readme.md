## Dataset:

### Cylinder wake

#### (1) Governing equation: 

* Incompressible

* Two-dimensional

* Domain: $-5<x<25$ and $-5<y<5$

* Reynolds number: 100

* Boundaries:

  * inlet: uniform 1
  * outlet: outflow
  * lateral: symmetric

* Initial condition: unstable steady solution form selective frequency damping method

* Observable of dataset: vorticity $\omega$
  $$
  \omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
  $$

#### (2) Solver: Nek5000

* Number of element: 2164

* Spectral order: 7

#### (3) Periodic cylinder wake

* Dataset: `periodic.dat`
  * Data size: $1000\times 192\times 384$

* Sampling parameters: 
  * domain: $-0.8\le x\le 8.8$ and $-2.4\le y \le 2.4$
  * size: $N_x\times N_y = 192\times 384$
  * time: $\Delta t = 0.2$
  * time period: $300\le T\le 500$ 

#### (4) Transient cylinder wake

* Dataset: `transient.dat`
  * Data size: $1500\times 384\times 768$

* Sampling parameters: 
  * domain: $-4.6\le x\le 14.6$ and $-4.8\le y \le 4.8$
  * size: $N_x\times N_y = 384\times 768$
  * time: $\Delta t = 0.1$
  * time period: $0<T<150$




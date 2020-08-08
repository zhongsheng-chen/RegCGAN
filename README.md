# RegCGAN 

Our RegCGAN is a new generator for producing virtual samples. 
Inspiring by CGAN, RegCGAN implicitly capture the condition 
probability `p(y|x)` from historical data as like CGAN doing for
`p(x|y)`.So, similar to CGAN, our RegCGAN is mainly made up of
a Generator *G* and a Discriminator *D*. The Generator *G* 
consumes *x* and *z* and yields fake *y*, while 
the Discriminator *D* consumes *x* and *y* and distinguish between
the true y and the fake one. Once RegCGAN is well trained,
it can serve as a generative probability model `p(y|x)`, just like 
Gaussian process (GP). Thanks to such ability, we make an attempt to
use it to synthesis output of any targeting points in the input space
For small sample size (SSS) soft modeling problems, data at hand suffers 
from data sparsity, leading to degrading performance of a soft model.
To handle this issue, we intend to create new points to fill up such
 areas of data sparsity in the input space through CVT sampling. The
 data sparsity regions is identified by Local Outlier Factor. The
 output of uniformly distributed new samples is synthesis by averaging
 a number of samples drawn from p(y|x), which is captured by RegCGAN.
 Because the generated samples has a similar behavior to the
 real samples when used to training a soft model,
  we call them virtual samples.

# Dependencies
This code requires the following package referenced in `requirements.txt`:
  * numpy, jupyter, matplotlib, pandas, seaborn
  * sklearn
  * tensorflow, keras
  * density, available at `https://pypi.org/project/diversipy/`
  * idaes-pse, available at `https://idaes-pse.readthedocs.io/en/stable/getting_started.html/`
  * For GPy, one can visit `https://sheffieldml.github.io/GPy/`
  
 all packages need to be installed on a conda environment with python >= 3.0 

# Getting started
* First, install must-have packages in the environment
 `pip install -r requirements.txt`. 
* Second, install RegCGAN itself by 
 typing `pip install -e .` at the root.
* Finally, run an applications in the notebooks using `jupyter-notebook`.

# Acknowledgement
We appreciate efforts in `https://github.com/eriklindernoren/Keras-GAN` and
in `https://github.com/mkirchmeyer/ganRegression`.
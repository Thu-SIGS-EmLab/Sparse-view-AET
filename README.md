# Generative model-based sparse-view atomic electron tomography
The NSIRE (Null-Space Interative REconstruction) package performs iterative sampling from the data distribution of atomic potential using a diffusion model until atomic tomogram that subject to the ADF-STEM projection constraints are obtained, enabling high accuracy (<20pm RMSD and ~0.1 R1 factor error) three-dimensional atomic structure reconstruction under sparse-view (10 to 20 projections), assists the AET reconstruction of irradiated sensitive samples and ultra-small nanoparticles.
![image](https://github.com/LIHAN8099/Sparse-view-AET/blob/main/NSIRE_01.png)

Through range-null space decomposition and ancestor sampling, the posterior mean x_0|t corresponding to each noisy tomogram x_t is gradually guided to the region of the atomic manifold satisfying the projection constraint y=Ax.

<p align="center">
  <img src=https://github.com/LIHAN8099/Sparse-view-AET/blob/main/rec.gif width="1500" alt="Iterative sampling process visualization"/>
</p>



# Requirements
- pytorch=1.10.1
- numpy
- scipy
- opencv-python
- torchvision

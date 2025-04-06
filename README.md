# Sparse-view-AET
The NSIRE (Null-Space Interative REconstruction) package performs iterative sampling from the data distribution of atomic potential using a diffusion model until atomic tomogram that subject to the ADF-STEM projection constraints are obtained, enabling high accuracy (<20pm RMSD and ~0.1 R1 factor error) three-dimensional atomic structure reconstruction under sparse-view (10 to 20 projections), assists AET reconstruction of irradiated sensitive samples and ultra-small nanoparticles.
![image](https://github.com/LIHAN8099/Sparse-view-AET/blob/main/NSIRE_01.png)

# Example: Spaese AET reconstruction of PtCo
Through range-null space decomposition and ancestor sampling, the posterior mean x_0|t corresponding to each noisy tomogram xt is gradually guided to the region of the atomic manifold satisfying the projection constraint y=Ax.

![image](https://github.com/LIHAN8099/Sparse-view-AET/blob/main/rec.gif)



# Requirements
- numpy
- pytorch=1.10.1
- scipy
- opencv-python
- tqdm

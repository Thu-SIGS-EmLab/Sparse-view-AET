# Sparse-view-AET
The NSIRE package performs iterative sampling from the data distribution of atomic potential using a diffusion model until atomic tomogram that subject to the ADF-STEM projection constraints are obtained, enabling high accuracy (<20pm RMSD and ~0.1 R1 factor error) three-dimensional atomic structure reconstruction under sparse-view (10 to 20 projections), assists AET reconstruction of irradiated sensitive samples and ultra-small nanoparticles.
![NSIRE_01](https://github.com/user-attachments/assets/63e948f2-dbb8-4085-942d-16e6a6192aa6)

# Example: Spaese AET reconstruction of PtCo
Through range-null space decomposition and ancestor sampling, the posterior mean x_0|t corresponding to each noisy tomogram xt is gradually guided to the region of the atomic manifold satisfying the projection constraint y=Ax.

![image](https://github.com/LIHAN8099/Sparse-view-AET/blob/main/rec.gif)



# Requirements
- numpy
- pytorch=1.10.1
- scipy
- opencv-python
- tqdm

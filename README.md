# Sparse-view atomic electron tomography

    The NSIRE (Null-Space Interative REconstruction) package performs iterative sampling from the data distribution of atomic potential using a diffusion model until atomic tomogram that subject to the ADF-STEM projection constraints are obtained, enabling high accuracy (<20pm RMSD and ~0.1 R1 factor error) three-dimensional atomic structure reconstruction under sparse-view (10 to 20 projections), assists the AET reconstruction of irradiated sensitive samples and ultra-small nanoparticles.
![image](https://github.com/LIHAN8099/Sparse-view-AET/blob/main/NSIRE_01.png)

    Through range-null space decomposition and ancestral sampling, the posterior mean x_0|t corresponding to each noisy tomogram x_t is gradually guided to the region of the atomic manifold satisfying the projection constraint y=Ax. Iterative sampling process visualization:

<p align="center">
  <img src=https://github.com/LIHAN8099/Sparse-view-AET/blob/main/rec.gif width="900"/>
</p>



# Requirements
- pytorch=1.10.0
- numpy
- scipy
- opencv-python

# Note
- The default pixel size of the pre-trained model is 0.3105 angstroms. When your experimental data is of other values, it is recommended to adjust it to around 0.3105 through interpolation.
- Please adjust the horizontal size of the projected data to 256 through cropping or padding.
- When phenomena such as stacking discontinuity or "atomic ejection" occur, please recheck the alignment or noise reduction, or increase the number of projections.
  
# Acknowledgements
This implementation is inspired by:
- Diffusion model: https://github.com/ermongroup/ddim, https://github.com/wyhuai/DDNM
- Atomic Electron Tomography:https://github.com/AET-AmorphousMaterials/Supplementary-Data-Codes, https://github.com/MDAIL-KAIST/DL-augmentation

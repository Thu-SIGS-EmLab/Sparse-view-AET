# Sparse-view atomic electron tomography
The NSIRE (Null-Space Interative REconstruction) package performs iterative sampling from the data distribution of atomic potential using a diffusion model until atomic tomogram that subject to the ADF-STEM projection constraints are obtained, enabling high accuracy (<20pm RMSD and ~0.1 R1 factor error) three-dimensional atomic structure reconstruction under sparse-view (10 to 20 projections) or limited-angle (within ±30°), assists the AET reconstruction of irradiated sensitive samples and ultra-small nanoparticles.
![image](https://github.com/LIHAN8099/Sparse-view-AET/blob/main/NSIRE_01.png)

Through range-null space decomposition and ancestral sampling, the posterior mean x_0|t corresponding to each noisy tomogram x_t is gradually guided to the region of the atomic manifold satisfying the projection constraint y=Ax. Iterative sampling process visualization:

<p align="center">
  <img src=https://github.com/LIHAN8099/Sparse-view-AET/blob/main/rec.gif width="900"/>
</p>



# Requirements
- pytorch>=1.10.0
- numpy
- scipy
- opencv-python

# Note
- When phenomena such as stacking discontinuity or "atomic distortion" occur, please recheck the alignment or noise reduction, or increase the number of projections.
- Please adjust the horizontal size of the projected tilt-series of nanoparticles to 256 through cropping or padding.
- The default pixel size of the pre-trained model is 31.05 pm. When your experimental data is of other values, it is recommended to adjust it to around 31.05 through interpolation.

# Run AET reconstruction

for small nanoparticle (256 pixels):
- python main_256.py --ni --config tomography256.yml --eta 0.85 --sigma_y 0.05 --ckpt "log/Pt_potential256.pt" -i Pt_LA --data_dir "examples/Pt_LA_exp/"
- python main_256.py --ni --config tomography256.yml --eta 0.85 --sigma_y 0.05 --ckpt "log/Pt_potential256.pt" -i MoS2 --data_dir "examples/MoS2/"

for large-sized nanoparticle (384 pixels):

- python main_384.py --ni --config tomography384.yml --eta 0.85 --sigma_y 0.05 --ckpt "log/Pt_potential384.pt" -i glass --data_dir "examples/glass/"
- python main_384.py --ni --config tomography384.yml --eta 0.85 --sigma_y 0.05 --ckpt "log/Pt_potential384.pt" -i Zr --data_dir "examples/Zr/"

for thin-film specimens:
- python main_film.py --ni --config tomography256.yml --eta 0.85 --sigma_y 0.05 --ckpt "log/Pt_potential256.pt" -i Ta --data_dir "examples/Ta/"

# Train your new score network

- python training_pre.py 
- mpirun -n 8 python image_train.py --data_dir ./training_set/ --image_size 64 --num_channels 64 --num_res_blocks 3 --learn_sigma False --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --microbatch 128


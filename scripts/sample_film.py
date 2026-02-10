import os
import logging
import time
import glob
import numpy as np
import tqdm
import torch.utils.data as data
import torchvision.utils as tvu
from improved_diffusion.script_util import create_model
import random
from scipy.linalg import orth
from tqdm import tqdm
import cv2
from scipy.sparse import coo_matrix
import torch
import scipy.io as sio
import math

device = torch.device("cuda")


def calc_A(angles, projections):  # Calculate the matrix of Radon transform
    dtype = 'float32'
    projections = projections.astype(dtype)
    dimx, dimy, Num_pj = projections.shape
    obj_dimx, obj_dimy, obj_dimz = dimx, dimy, dimx
    ncx = round((dimx + 1) / 2)
    ncx_ext = np.ceil((obj_dimx + 1) / 2)
    ZZ, XX = np.meshgrid(np.arange(1, obj_dimz + 1) - ncx_ext, np.arange(1, obj_dimx + 1) - ncx_ext)
    XX, ZZ = XX.flatten(), ZZ.flatten()
    pt_o_ratio = 4
    num_pts = XX.size
    vec_ind = np.arange(1, num_pts + 1)
    rot_mat_k = [coo_matrix((dimx, num_pts), dtype=dtype) for _ in range(Num_pj)]

    for k in range(0, Num_pj):
        theta = angles[k]
        R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                      [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        rot_x_o = (R[0] @ np.vstack([ZZ, XX])) + ncx
        for s in range(1, pt_o_ratio + 1):
            s1, s2 = 1, 1
            rot_x_shift = R[0] @ np.vstack([(-1) ** s1, (-1) ** s2]) * 0.25
            rot_x = rot_x_o + rot_x_shift
            x_foor = np.floor(rot_x).astype(int)
            goodInd = (x_foor >= 1) & (x_foor < dimx)
            vec_goodInd1 = vec_ind[goodInd]
            x1 = x_foor[goodInd]
            b1 = x1 + 1 - rot_x[goodInd]
            goodInd = (x_foor == 0)
            vec_goodInd2 = vec_ind[goodInd]
            x2 = x_foor[goodInd] + 1
            b2 = rot_x[goodInd]
            goodInd = (x_foor == dimx)
            vec_goodInd3 = vec_ind[goodInd]
            x3 = x_foor[goodInd]
            b3 = 1 + dimx - rot_x[goodInd]
            masterSub = np.column_stack([np.concatenate([x1, x1 + 1, x2, x3]),
                                         np.concatenate([vec_goodInd1, vec_goodInd1, vec_goodInd2, vec_goodInd3])])
            masterVal = np.concatenate([b1, 1 - b1, b2, b3])
            rot_mat_k[k] += coo_matrix((masterVal, (masterSub[:, 0] - 1, masterSub[:, 1] - 1)),
                                       shape=(dimx, num_pts), dtype=dtype)
    A = np.vstack([rot_mat.todense() / pt_o_ratio for rot_mat in rot_mat_k])
    nonzero_coords = np.transpose(np.nonzero(A))
    values = A[nonzero_coords[:, 0], nonzero_coords[:, 1]]
    indices = torch.tensor(nonzero_coords, dtype=torch.long).t()
    values = torch.tensor(values, dtype=torch.float32).squeeze()
    size = torch.Size([A.shape[0], A.shape[1]])
    A = torch.sparse.FloatTensor(indices, values, size)
    A = A.to(device)
    return A


def A_pinv(A, y, iterations=50):  # Calculate the inverse Radon transform iteratively
    step_size = 2
    dimx, dimy, Num_pj = y.shape
    dt = torch.tensor((step_size / Num_pj / dimx), dtype=torch.float32)
    projections = y
    projections_vec = projections.permute(2, 0, 1).reshape(dimx * Num_pj, dimy)
    projections_vec = projections_vec.to(device)
    rec_vec = torch.zeros((dimx * dimx, dimy))
    rec_vec = rec_vec.to(device)
    # Tomography reconstruction: solve Ax=y
    for iter in range(iterations):
        pj_cals = torch.sparse.mm(A, rec_vec)
        residual = pj_cals - projections_vec
        grad = torch.sparse.mm(A.t(), residual)
        rec_vec = rec_vec - dt * grad
    rec = rec_vec.reshape((dimx, dimx, dimy))
    return rec


def Ap(angles, projections, iterations):  # Calculate the matrix of inverse Radon transform based on RESIRE
    dtype = 'float32'
    projections = projections.astype(dtype)
    dimx, dimy, Num_pj = projections.shape
    obj_dimx, obj_dimy, obj_dimz = dimx, dimy, dimx
    ncx = round((dimx + 1) / 2)
    ncx_ext = np.ceil((obj_dimx + 1) / 2)
    ZZ, XX = np.meshgrid(np.arange(1, obj_dimz + 1) - ncx_ext, np.arange(1, obj_dimx + 1) - ncx_ext)
    XX, ZZ = XX.flatten(), ZZ.flatten()
    pt_o_ratio = 4
    num_pts = XX.size
    vec_ind = np.arange(1, num_pts + 1)
    rot_mat_k = [coo_matrix((dimx, num_pts), dtype=dtype) for _ in range(Num_pj)]

    for k in range(0, Num_pj):
        theta = angles[k]
        R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                      [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
        rot_x_o = (R[0] @ np.vstack([ZZ, XX])) + ncx
        for s in range(1, pt_o_ratio + 1):
            s1, s2 = 1, 1
            rot_x_shift = R[0] @ np.vstack([(-1) ** s1, (-1) ** s2]) * 0.25
            rot_x = rot_x_o + rot_x_shift
            x_foor = np.floor(rot_x).astype(int)
            goodInd = (x_foor >= 1) & (x_foor < dimx)
            vec_goodInd1 = vec_ind[goodInd]
            x1 = x_foor[goodInd]
            b1 = x1 + 1 - rot_x[goodInd]
            goodInd = (x_foor == 0)
            vec_goodInd2 = vec_ind[goodInd]
            x2 = x_foor[goodInd] + 1
            b2 = rot_x[goodInd]
            goodInd = (x_foor == dimx)
            vec_goodInd3 = vec_ind[goodInd]
            x3 = x_foor[goodInd]
            b3 = 1 + dimx - rot_x[goodInd]
            masterSub = np.column_stack([np.concatenate([x1, x1 + 1, x2, x3]),
                                         np.concatenate([vec_goodInd1, vec_goodInd1, vec_goodInd2, vec_goodInd3])])
            masterVal = np.concatenate([b1, 1 - b1, b2, b3])
            rot_mat_k[k] += coo_matrix((masterVal, (masterSub[:, 0] - 1, masterSub[:, 1] - 1)),
                                       shape=(dimx, num_pts), dtype=dtype)
    A = torch.tensor(np.vstack([rot_mat.todense() / pt_o_ratio for rot_mat in rot_mat_k])).to(device)

    step_size = 2
    dt = torch.tensor((step_size / Num_pj / dimx), dtype=torch.float32)
    ATA = torch.matmul(A.T, A).to(device)
    Apinv = torch.zeros_like(ATA)
    I = torch.eye(ATA.shape[0]).to(device)
    I_minus_ATA = I - dt * ATA
    for n in range(iterations):
        print(n)
        Apinv = Apinv + dt * torch.linalg.matrix_power(I_minus_ATA, n)
    Apinv = torch.matmul(Apinv, A.T)
    return Apinv


def scale_pj(A, y, path):
    Apy = A_pinv(A, y, 100)
    As = Apy.shape
    A0 = Apy.cpu().squeeze().detach().numpy()
    
    sio.savemat("%s" % path, {"Apy": A0 / np.max(A0) * 255})
    Apy = Apy / torch.max(Apy)
    y_hat = torch.sparse.mm(A, Apy.reshape(As[0] * As[1], As[2]))
    y = (y / torch.sum(y)) * torch.sum(y_hat)
    return y


def sAET(projections, angles, subvol_size, r_o): # scanning AET
    sub_projs = []
    angles = angles * math.pi / 180
    final_subvol_size = np.ceil(subvol_size * np.sqrt(2) / 4)
    Nx, Ny, Nangles = projections.shape
    target_size = (subvol_size / 4, Ny)
    r_vx = subvol_size / Nx
    M_x = math.ceil((1 - r_vx * r_o) / (r_vx - r_vx * r_o))
    x_centers = []
    for m in range(1, M_x + 1):
        x_m = int((Nx / 2) * (1 + (r_vx - r_o * r_vx) * (2 * m - M_x - 1)))
        x_centers.append(x_m)
    VOL = np.zeros((subvol_size, projections.shape[0], projections.shape[1]))
    VOLG = np.ones_like(VOL) * 10 ** (-6)
    mask = np.zeros((subvol_size, projections.shape[1], angles.shape[0]))
    k = 0
    for i in angles:
        l = np.ceil(subvol_size * np.cos(i))
        pad = int((subvol_size - l) / 2)
        mask[pad:int(subvol_size - pad), :, k] = 1
        k += 1
    for x_center in x_centers:
        x_offset = x_center - Nx / 2
        sub_proj = np.zeros((subvol_size, Ny, Nangles), dtype=projections.dtype)
        for k in range(Nangles):
            theta = angles[k]
            required_shift = -x_offset * math.cos(theta)
            current_proj = projections[:, :, k].copy()
            shift_x = required_shift
            ky = np.fft.fftfreq(Ny)
            kx = np.fft.fftfreq(Nx)
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            phase_shift = np.exp(-2j * np.pi * kx_grid * shift_x)
            fft_proj = np.fft.fft2(current_proj)
            shifted_fft = fft_proj * phase_shift
            shifted_proj = np.real(np.fft.ifft2(shifted_fft))
            center_x = Nx // 2
            crop_start_x = center_x - subvol_size // 2
            crop_end_x = crop_start_x + subvol_size
            if crop_start_x < 0:
                pad_left = -crop_start_x
                crop_start_x = 0
            else:
                pad_left = 0
            if crop_end_x > Nx:
                pad_right = crop_end_x - Nx
                crop_end_x = Nx
            else:
                pad_right = 0
            cropped = shifted_proj[crop_start_x:crop_end_x, :]
            if pad_left > 0 or pad_right > 0:
                cropped = np.pad(cropped, ((pad_left, pad_right), (0, 0)),
                                 mode='constant')
            sub_proj[:, :, k] = cropped[:subvol_size, :]
        sub_proj = sub_proj * mask
        sub_proj[sub_proj < 0] = 0
        sub_projs.append(sub_proj)
        
    if projections.shape[0]==subvol_size:
        sub_projs=[projections* mask]

    return sub_projs, final_subvol_size, target_size, VOL, VOLG, x_centers


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"))

        self.device = device

        self.model_var_type = config.diffusion.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        ckpt = self.args.ckpt
        print(ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        model = torch.nn.DataParallel(model)

        print('Run Tomography Reconstruction',
              f'{self.config.time_travel.T_sampling} sampling steps.')
        self.run(model)

    def run(self, model):
        args, config = self.args, self.config
        g = torch.Generator()
        g.manual_seed(args.seed)

        dir = args.data_dir
        proj_all = np.load(os.path.join(dir, "projections.npy"))
        angles = np.loadtxt(os.path.join(dir, "Angles.txt"))

        print(angles)

        subvol_size = 256
        r_o = 0.45
        if proj_all.shape[0]!=subvol_size:
            proj_all = padding(subvol_size // 2, proj_all)
        raw_proj = proj_all / np.max(proj_all)
        raw_proj = raw_proj.astype(np.float32)
        A = calc_A(angles, raw_proj)
        raw_proj = torch.tensor(raw_proj, device=self.device)
        yn = scale_pj(A, raw_proj, os.path.join(self.args.image_folder, f"Apy.mat"))
        yn = yn.cpu().numpy()
        
        sigma_y = 2 * args.sigma_y

        sub_projs, final_subvol_size, target_size, VOL_APY, VOLG, x_centers = sAET(yn, angles, subvol_size, r_o)
        
        VOL_NSIRE = np.zeros_like(VOL_APY)

        for numpj in range(len(sub_projs)):
            x_center = x_centers[numpj]
            raw_proj = sub_projs[numpj]
            np.save(os.path.join(self.args.image_folder, f"scan_proj_%s.npy"%numpj),raw_proj)
            if numpj == 0:
                A = calc_A(angles, raw_proj)
                Y_list = start_points(raw_proj.shape[1], 8, 0.25)
                gaussian_map = get_gaussian((subvol_size, subvol_size, 8))

            new_vol = torch.zeros((raw_proj.shape[0], raw_proj.shape[0], raw_proj.shape[1]))
            normalization = torch.zeros(new_vol.shape, dtype=torch.float32)

            raw_proj = torch.tensor(raw_proj, device=self.device,dtype=torch.float32)

            Apy = A_pinv(A, raw_proj, 100)
            A0 = Apy.cpu().squeeze().detach().numpy()
            A0[A0<0]=0
            sio.savemat(os.path.join(self.args.image_folder, f"Apy_%s.mat") % numpj, {"Apy": A0})

            if int(x_center - final_subvol_size) < 0:
                final_subvol_size = x_center
            elif int(x_center + final_subvol_size) > proj_all.shape[0]:
                final_subvol_size = np.round(proj_all.shape[0] - x_center)
            else:
                pass

            sub_vol = A0[:,int(subvol_size / 2 - final_subvol_size):int(subvol_size / 2 + final_subvol_size), :]
            g = get_gaussian(sub_vol.shape).cpu().numpy()
            VOL_APY[:,int(x_center - final_subvol_size):int(x_center + final_subvol_size), :] += sub_vol * g
            VOLG[:,int(x_center - final_subvol_size):int(x_center + final_subvol_size), :] += g

            x_mins_one = None
            
            for Y in Y_list:
                print(x_center, Y)
                y = raw_proj[:, Y:Y + 8, :]
            
                B = 1
                x = torch.randn(B, config.data.channels, config.data.image_size,
                                config.data.image_size,
                                device=self.device)
                with torch.no_grad():
                    skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
                    n = x.size(0)
                    x0_preds = []
                    xs = [x]
            
                    times = get_schedule_jump(config.time_travel.T_sampling,
                                              config.time_travel.travel_length,
                                              config.time_travel.travel_repeat,
                                              )
                    time_pairs = list(zip(times[:-1], times[1:]))
            
                    for i, j in tqdm(time_pairs):
                        if j > -10:
                            i, j = i * skip, j * skip
                            if j < 0:
                                j = -1
                            if j < i:
                                t = (torch.ones(n) * i).to(x.device)
                                next_t = (torch.ones(n) * j).to(x.device)
                                at = compute_alpha(self.betas, t.long())
                                at_next = compute_alpha(self.betas, next_t.long())
                                sigma_t = (1 - at_next ** 2).sqrt()
                                xt = xs[-1].to('cuda')
                                et = model(xt, t)
            
                                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                                # if x_mins_one is not None:
                                #     x0_t[:, :2, :, :] = x_mins_one
            
                                if sigma_t >= at_next * sigma_y:
                                    lambda_t = 1.
                                    gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                                else:
                                    lambda_t = (sigma_t) / (at_next * sigma_y)
                                    gamma_t = 0.

            
                                Ax0_t = torch.sparse.mm(A, (x0_t).squeeze().permute(1, 2, 0).reshape(subvol_size * subvol_size, 8))
                                res = Ax0_t.reshape((y.shape[2], y.shape[0], y.shape[1])).permute(1, 2, 0) - y
                                Apr = A_pinv(A, res).permute(2, 0, 1).unsqueeze(0)
                                
                                x0_t_hat = x0_t - lambda_t * Apr
            
                                eta = self.args.eta
            
                                c1 = (1 - at_next).sqrt() * eta
                                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            
                                xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (
                                        c1 * torch.randn_like(x0_t) + c2 * et)
            
                                x0_preds.append(x0_t.to('cpu'))
                                xs.append(xt_next.to('cpu'))
            
                            else:
                                next_t = (torch.ones(n) * j).to(x.device)
                                at_next = compute_alpha(self.betas, next_t.long())
                                x0_t = x0_preds[-1].to('cuda')
            
                                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
            
                                xs.append(xt_next.to('cpu'))
            
                            if j == 0:
                                NOV = xt_next.to('cpu')
            
                    x = xs[-1]
            
                x = [xi for xi in x]
                # x_mins_one = x[0][6:, :, :].to('cuda')
                new_vol[:, :, Y:Y + 8] += x[0].squeeze().permute(1, 2, 0) * gaussian_map
                normalization[:, :, Y:Y + 8] += gaussian_map
                tomogram = NOV.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                tomogram = tomogram / np.max(tomogram) * 255
                tomogram = tomogram[:, :, 7]
                cv2.imwrite(os.path.join(self.args.image_folder, f"{Y}_{2}.png"),
                            tomogram)
            new_vol = new_vol.to(self.device) / normalization.to(self.device)
            new_rec = new_vol.cpu().squeeze().detach().cpu().numpy()
            sio.savemat(os.path.join(self.args.image_folder, f"rec%s.mat"%numpj), {"rec": new_rec / np.max(new_rec) * 255})
            sub_vol_ns = new_rec[:,int(subvol_size / 2 - final_subvol_size):int(subvol_size / 2 + final_subvol_size), :]
            VOL_NSIRE[:,int(x_center - final_subvol_size):int(x_center + final_subvol_size), :] += sub_vol_ns * g

        VOL_APY /= VOLG
        VOL_NSIRE /= VOLG

        sio.savemat(os.path.join(self.args.image_folder, f"Apy.mat"), {"rec": VOL_APY / np.max(VOL_APY) * 255})
        sio.savemat(os.path.join(self.args.image_folder, f"rec.mat"), {"rec": VOL_NSIRE / np.max(VOL_NSIRE) * 255})


# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts


def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        import math
        betas = betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def LOG(iter, data):
    data = data.to("cpu")
    img = data.squeeze().permute(1, 2, 0)[:, :, 4].numpy()
    img = img / np.max(img) * 255
    cv2.imwrite("LOG_xt/%s.png" % iter, img)
    return


def LOG1(iter, data):
    data = data.to("cpu")
    img = data.squeeze().permute(1, 2, 0)[:, :, 4].numpy()
    img = img / np.max(img) * 255
    cv2.imwrite("LOG_x0t/%s.png" % iter, img)
    return


def circle(size, radius):
    c = torch.ones((size, size))
    C = torch.zeros_like(c)
    center_x = center_y = size // 2
    for y in range(size):
        for x in range(size):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                C[y, x] = c[y, x]
    return C.cuda()


C = circle(256, 128)


def get_gaussian(s, sigma=1.0 / 8):
    temp = np.zeros(s)
    coords = [i // 2 for i in s]
    sigmas = [i * sigma for i in s]
    temp[tuple(coords)] = 1
    from scipy.ndimage import gaussian_filter
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode='constant', cval=0)
    gaussian_map /= np.max(gaussian_map)
    gaussian_map = torch.tensor(gaussian_map, dtype=torch.float32)
    return gaussian_map


def padding(size, pj):
    new = np.zeros((pj.shape[0] + size, pj.shape[1], pj.shape[2]))
    new[int(size // 2):int(size // 2 + pj.shape[0]), :, :] = pj
    return new


def start_points(size, split_size, overlap):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

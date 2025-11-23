'''
The empirical multi-energy calibration with forward projection
'''

# %%
import SimpleITK as sitk
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from typing import List
from leapctype import tomographicModels

import pcct_md_diffusion.utils as utils
from pcct_md_diffusion.locations import base_input_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seg_filename', default='segmentation.seg.nrrd')
    parser.add_argument('--seg_mat_filename', default='materials.csv')
    parser.add_argument('--mono_mat_filename', default='omnitom_pcd/calibration/mono_att_coef_nist_mix.csv')
    parser.add_argument('--geometry', default='omnitom_pcd/geometry/omnitom_6x5_parallel_fov_308.cfg')

    parser.add_argument('--target', default='70', help='Decomposition target')
    parser.add_argument('--exclusion', default='', help='The ROI to be excluded from the target')
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--order_base', type=float, default=1)

    parser.add_argument('--offset', type=float, default=1000)
    parser.add_argument('--norm', type=float, default=1000 / 0.02)

    parser.add_argument('--device', type=int, default=0)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = 1
    else:
        args = parser.parse_args()
        args.debug = 0

    args = utils.get_run_info(args)

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def get_decomposition_target_mono(
    seg_filename,
    seg_mat_filename,
    mono_mat_filename,
    target_energy: int,
    segs_to_exclude: List[int],
):
    '''
    Generate the decomposition target image and mask
    '''
    # material information
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_filename))
    manifest_mat = pd.read_csv(seg_mat_filename)
    manifest_mono = pd.read_csv(mono_mat_filename)

    # only select the energy of interest
    manifest_mono = manifest_mono[manifest_mono['energy'] == int(target_energy)]

    # remove the segs to be excluded
    manifest_mat = manifest_mat[['Seg', 'Material']]
    manifest_mat = manifest_mat[~manifest_mat['Seg'].isin(segs_to_exclude)]
    manifest_mat['Target'] = None

    # assign the target value for each seg
    for i, row in manifest_mat.iterrows():
        if row['Material'] in manifest_mono:
            manifest_mat.at[i, 'Target'] = manifest_mono[row['Material']].values[0]

    # remove the segs that does not have any value
    manifest_mat = manifest_mat[~manifest_mat['Target'].isna()]

    # assign the values to the segmentation image
    mask = np.zeros_like(seg)
    target = np.zeros(seg.shape, np.float32)
    for i, row in manifest_mat.iterrows():
        inds = np.where(seg == row['Seg'])
        mask[inds] = 1
        target[inds] = row['Target'] / 10  # convert from cm-1 to mm-1

    # get the water value under the target
    water_val = manifest_mono['true_water'].values[0] / 10

    return target, mask, water_val


# def get_decomposition_target_basis(args):
#     '''
#     Generate the decomposition target image and mask
#     '''
#     # material information
#     seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, args.seg_filename)))
#     # seg = seg[:, ::-1, :]
#     manifest_mat = pd.read_csv(os.path.join(working_dir, args.seg_mat_filename))

#     # remove the segs to be excluded
#     manifest_mat = manifest_mat[['Seg', 'Material', args.target]]
#     segs_exclude = [int(s) for s in args.exclusion.split(',')]
#     manifest_mat = manifest_mat[~manifest_mat['Seg'].isin(segs_exclude)]

#     # assign the target value for each seg
#     manifest_mat['Target'] = manifest_mat[args.target].copy()

#     # remove the segs that does not have any value
#     manifest_mat = manifest_mat[~manifest_mat['Target'].isna()]

#     # assign the values to the segmentation image
#     mask = np.zeros_like(seg)
#     target = np.zeros(seg.shape, np.float32)
#     for i, row in manifest_mat.iterrows():
#         inds = np.where(seg == row['Seg'])
#         mask[inds] = 1
#         target[inds] = row['Target']

#     # get the water value under the target
#     water_val = 1

#     return target, mask, water_val


# %%
def fit_prj_to_decomp(
    prjs: np.array,
    decomp_target: np.array,
    decomp_mask: np.array,
    leapct: tomographicModels,
    order: int,
    order_base: float
):
    order_list = []
    for poly_orders in itertools.product(range(order + 1), repeat=prjs.shape[0]):
        poly_orders = np.array(poly_orders)
        if np.sum(poly_orders) == 0:
            continue
        if np.sum(poly_orders) > order:
            continue
        order_list.append(poly_orders * order_base)

    poly_recons = []
    print(len(order_list))
    for i, poly_orders in enumerate(order_list):
        print(i, end=',', flush=True)

        poly_prjs = 1
        for prj, prj_order in zip(prjs, poly_orders):
            poly_prjs *= prj**prj_order
        poly_prjs = np.copy(poly_prjs, 'C').astype(np.float32)
        fbp = leapct.FBP(poly_prjs)
        poly_recons.append(fbp)

    # reverse weight by number of pixels
    y = decomp_target[np.where(decomp_mask)]
    w = np.ones_like(y)
    # vals, cnts = np.unique(y, return_counts=True)
    # for v, cnt in zip(vals, cnts):
    #     w[y == v] = 1 / cnt

    # poly fit inside the mask
    y = decomp_target[np.where(decomp_mask)] * w
    mat = []
    for recon in poly_recons:
        mat.append(recon[np.where(decomp_mask)] * w)
    mat = np.array(mat)
    print('Condition number', np.linalg.cond(mat))
    coefs = np.linalg.lstsq(mat.T, y, rcond=None)[0]

    return coefs, np.array(order_list)


# %%
def emec_with_coefs(
    prjs: np.array,
    coefs: np.array,
    order_list: np.array,
    leapct: tomographicModels,
):
    # combine in the projection domain
    decomp_prjs = 0
    for coef, poly_orders in zip(coefs, order_list):
        poly_prjs = 1
        for prj, prj_order in zip(prjs, poly_orders):
            poly_prjs *= prj**prj_order
        decomp_prjs += coef * poly_prjs
    decomp_prjs = np.copy(decomp_prjs, 'C').astype(np.float32)

    fbp = leapct.FBP(decomp_prjs)

    return fbp


# %%
def main(args):
    input_dir = os.path.join(base_input_dir, args.input_dir)
    output_dir = os.path.join(base_input_dir, args.output_dir)
    mono_mat_filename = os.path.join(base_input_dir, args.mono_mat_filename)
    geometry_filename = os.path.join(base_input_dir, args.geometry)
    os.makedirs(output_dir, exist_ok=True)

    segs_to_exclude = [int(s) for s in args.exclusion.split(',') if len(s) > 0]

    imgs, dx, dy, dz = utils.load_image(input_dir)
    imgs[imgs == -1224] = -1000  # out of FOV
    # convert to attenuation coefficient
    imgs = (imgs + args.offset) / args.norm
    imgs = imgs.astype(np.float32)

    # read the decomposition target
    try:
        kev = int(args.target)
        print('Decomposition target is mono', kev)
        decomp_target, decomp_mask, water_val = get_decomposition_target_mono(
            os.path.join(input_dir, args.seg_filename),
            os.path.join(input_dir, args.seg_mat_filename),
            mono_mat_filename,
            kev,
            segs_to_exclude,
        )
    except Exception:
        print('Decomposition target is basis', args.target)
        raise NotImplementedError
        # decomp_target, decomp_mask, water_val = get_decomposition_target_basis(args)

    if args.debug:
        plt.figure(figsize=[12, 6])
        plt.subplot(121)
        plt.imshow(
            decomp_target[decomp_target.shape[0] // 2],
            'gray',
            vmin=0.84 * water_val,
            vmax=1.24 * water_val
        )
        plt.subplot(122)
        plt.imshow(decomp_mask[decomp_mask.shape[0] // 2], 'gray')
        plt.show()

    # Set the air to 0
    # Because we are using forward projection, we can manipulate the images as long as the assigned attenuation
    # coefficients are correct.
    inds = np.where((decomp_target == 0) & (decomp_mask == 1))
    for ic in range(imgs.shape[0]):
        # preserve some noise texture by subtracting the mean inside the mask
        imgs[ic][inds] -= np.mean(imgs[ic][inds])

    # forward projection
    print('Loading projector and geometry...', flush=True)
    geo = utils.load_geometry(geometry_filename)
    geo.nx = imgs.shape[3]
    geo.ny = imgs.shape[2]
    geo.nz = imgs.shape[1]
    geo.nv = geo.nz
    geo.dx = dx
    geo.dy = dy
    geo.dz = dz
    geo.dv = geo.dz

    leapct = utils.set_leapct_geometry(geo)
    leapct.print_parameters()

    print('Forward projection...', flush=True)
    prj = leapct.allocate_projections()
    prjs = []
    for ic in range(imgs.shape[0]):
        img = np.copy(imgs[ic], 'C').astype(np.float32)
        leapct.project(prj, img)
        prjs.append(np.copy(prj, 'C'))
    prjs = np.array(prjs, dtype=np.float32)

    # validate the forward projection
    fbps = []
    for ic in range(prjs.shape[0]):
        prj = np.copy(prjs[ic], 'C')
        fbp = leapct.FBP(prj)
        fbps.append(fbp)
    fbps = np.array(fbps, dtype=np.float32)
    if args.debug:
        plt.figure(figsize=[18, 6])
        plt.subplot(131)
        plt.imshow(imgs[0, imgs.shape[1] // 2], 'gray', vmin=0.84 * 0.02, vmax=1.24 * 0.02)
        plt.subplot(132)
        plt.imshow(fbps[0, fbps.shape[1] // 2], 'gray', vmin=0.84 * 0.02, vmax=1.24 * 0.02)
        plt.subplot(133)
        plt.imshow((imgs - fbps)[0, imgs.shape[1] // 2], 'gray', vmin=-0.01 * 0.02, vmax=0.01 * 0.02)
        plt.show()

    coefs, order_list = fit_prj_to_decomp(
        prjs, decomp_target, decomp_mask, leapct, args.order, args.order_base
    )

    print('Fitted coefficients:', coefs, flush=True)
    print('Order list:', order_list, flush=True)

    mono = emec_with_coefs(prjs, coefs, order_list, leapct)
    rmse_mask = np.where(decomp_target > 0, 1, 0)
    if args.debug:
        plt.figure(figsize=[18, 6])
        plt.subplot(131)
        plt.imshow(
            decomp_target[decomp_target.shape[0] // 2],
            'gray',
            vmin=0.84 * water_val,
            vmax=1.24 * water_val
        )
        plt.subplot(132)
        plt.imshow(mono[mono.shape[0] // 2], 'gray', vmin=0.84 * water_val, vmax=1.24 * water_val)
        plt.subplot(133)
        # plt.imshow(decomp_target[decomp_target.shape[0] // 2], 'gray', vmin=0.84 * water_val, vmax=1.24 * water_val)
        plt.imshow(
            np.abs((decomp_target - mono) * decomp_mask)[decomp_target.shape[0] // 2],
            'gray',
            vmin=0,
            vmax=0.1 * water_val
        )
        plt.show()
    # estimate decomposition error
    rmse_mask = np.where(decomp_target > 0, 1, 0)
    rmse = np.sqrt(np.sum((mono - decomp_target) ** 2 * rmse_mask) / np.sum(rmse_mask))
    print('RMSE on target', rmse / water_val * 1000)

    # save results
    output_manifest_filename = os.path.join(output_dir, 'fp_md_emec_{0}_order{1}.csv'.format(args.target, args.order))
    df_output = []
    for coef, poly_orders in zip(coefs, order_list):
        row = {'Target': args.target, 'WaterVal': water_val}
        for i, order in enumerate(poly_orders):
            row['OrderChannel{0}'.format(i)] = order
        row['Coefficient'] = coef
        df_output.append(row)
    df_output = pd.DataFrame(df_output)
    df_output.to_csv(output_manifest_filename, index=False)

    print('All done.', flush=True)

    return mono, df_output


# %%
if __name__ == '__main__':
    args = get_args([
        '--input_dir', 'omnitom_pcd/calibration/calibration_1_7/img/1',
        '--output_dir', 'omnitom_pcd/calibration/calibration_1_7/',
        '--order', '3',
        '--order_base', '1',
        '--target', '70',
        '--exclusion', '8,9,10,11',
    ])
    res = main(args)

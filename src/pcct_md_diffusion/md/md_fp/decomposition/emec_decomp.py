'''
Decomposition with the trained emec
'''

# %%
import SimpleITK as sitk
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from leapctype import tomographicModels

import pcct_md_diffusion.utils as utils
from pcct_md_diffusion.locations import base_input_dir, base_output_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_filename', required=True)
    parser.add_argument('--emec_coefs', required=True)
    parser.add_argument('--geometry', default='omnitom_pcd/geometry/omnitom_3x3_parallel_fov_252.cfg')

    parser.add_argument('--offset', type=float, default=1000)
    parser.add_argument('--norm', type=float, default=1000 / 0.02)

    parser.add_argument('--slice_average', type=int, default=1)

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
    print('Decomposed projection shape:', decomp_prjs.shape, flush=True)

    fbps = []
    for iv in range(decomp_prjs.shape[1]):
        if (iv + 1) % 10 == 0:
            print('Reconstructing image {}/{}'.format(iv + 1, decomp_prjs.shape[1]), flush=True)
        prj = np.copy(decomp_prjs[:, [iv]], 'C')
        fbp = leapct.FBP(prj)
        fbps.append(fbp)
    fbps = np.concatenate(fbps, axis=0)

    return fbps


# %%
def load_emec_coefs(filename):
    manifest = pd.read_csv(filename)

    water_val = manifest['WaterVal'].values[0]
    coefs = manifest['Coefficient'].values
    order_list = []
    names = sorted([c for c in manifest.columns if c.startswith('OrderChannel')])
    for name in names:
        order_list.append(manifest[name].values)
    order_list = np.array(order_list).T

    return coefs, order_list, water_val


# %%
def main(args):
    input_dir = os.path.join(base_input_dir, args.input_dir)
    emec_coefs_filename = os.path.join(base_input_dir, args.emec_coefs)
    geometry_filename = os.path.join(base_input_dir, args.geometry)
    output_filename = os.path.join(base_output_dir, args.output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Load the images
    imgs, dx, dy, dz, sitk_template = utils.load_image(input_dir, return_sitk_template=True)
    imgs[imgs == -1224] = -1000
    imgs = (imgs + args.offset) / args.norm
    imgs = imgs.astype(np.float32)
    # average slices if needed
    if args.slice_average > 1:
        nz = imgs.shape[1] // args.slice_average * args.slice_average
        imgs = imgs[:, :nz]
        imgs = imgs.reshape(imgs.shape[0], -1, args.slice_average, imgs.shape[2], imgs.shape[3]).mean(axis=2)
        dz = dz * args.slice_average
        spacing = list(sitk_template.GetSpacing())
        spacing = (spacing[0], spacing[1], dz)
        sitk_template.SetSpacing(spacing)
    print('Image shape:', imgs.shape, flush=True)

    # forward projection
    print('Load geometry...', flush=True)
    geo = utils.load_geometry(geometry_filename)
    geo.nx = imgs.shape[3]
    geo.ny = imgs.shape[2]
    geo.nz = 1
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
    for ch in range(imgs.shape[0]):
        print('Channel', ch, flush=True)
        ch_prjs = []
        for iz in range(imgs.shape[1]):
            img = np.copy(imgs[ch, [iz]], 'C')
            leapct.project(prj, img)
            ch_prjs.append(np.copy(prj, 'C'))
        prjs.append(np.concatenate(ch_prjs, axis=1))
    prjs = np.array(prjs)
    print('Projection shape:', prjs.shape, flush=True)

    # validate the forward projection
    prj = np.copy(prjs[0, :, prjs.shape[2] // 2], 'C')
    fbp = leapct.FBP(prj)
    if args.debug:
        plt.figure(figsize=[18, 6])
        plt.subplot(131)
        plt.imshow(imgs[0, imgs.shape[1] // 2], 'gray', vmin=0.02, vmax=0.022)
        plt.subplot(132)
        plt.imshow(fbp[fbp.shape[0] // 2], 'gray', vmin=0.02, vmax=0.022)
        plt.subplot(133)
        plt.imshow((imgs[0] - fbp)[imgs.shape[1] // 2], 'gray', vmin=-0.0002, vmax=0.0002)
        plt.show()

    coefs, order_list, water_val = load_emec_coefs(emec_coefs_filename)

    mono = emec_with_coefs(prjs, coefs, order_list, leapct)
    if args.debug:
        plt.figure(figsize=[12, 6])
        plt.subplot(121)
        plt.imshow(fbp[fbp.shape[0] // 2], 'gray', vmin=0.02, vmax=0.022)
        plt.subplot(122)
        plt.imshow(mono[mono.shape[0] // 2], 'gray', vmin=1 * water_val, vmax=1.1 * water_val)
        plt.show()

    mono = mono / water_val * 1000 - 1000
    mono = mono.astype(np.int16)
    sitk_res = sitk.GetImageFromArray(mono)
    sitk_res.SetSpacing(sitk_template.GetSpacing())
    sitk_res.SetOrigin(sitk_template.GetOrigin())
    sitk_res.SetDirection(sitk_template.GetDirection())
    sitk.WriteImage(sitk_res, output_filename)

    print('All done.', flush=True)

    return imgs, mono


# %%
if __name__ == '__main__':
    args = get_args([
        '--input_dir', 'omnitom_pcd/recon/img/3',
        '--output_filename', 'md_baseline/recon/img/3/fp_mmd_emec/mono_70_keV.nii.gz',
        '--emec_coefs', 'omnitom_pcd/calibration/calibration_1_7/fp_md_emec_70_order3.csv',
        '--slice_average', '6',
        '--geometry', 'omnitom_pcd/geometry/omnitom_6x5_parallel_fov_252.cfg',
    ])
    res = main(args)

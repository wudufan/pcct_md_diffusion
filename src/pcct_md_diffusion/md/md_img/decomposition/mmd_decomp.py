'''
Perform image domain multi-material decomposition
'''

# %%
import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

from typing import Tuple

import pcct_md_diffusion.utils as utils
from pcct_md_diffusion.locations import base_input_dir, base_output_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_scan', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--input_calib', required=True)
    parser.add_argument('--input_mono_att', default='omnitom_pcd/calibration/mono_att_coef_nist_mix.csv')
    parser.add_argument('--base_mat_0', default='blood_100')
    parser.add_argument('--base_mat_1', default='iodine_5')
    parser.add_argument(
        '--mmd_triplets',
        default='air,blood_nist,omnipaque_350_blood_nist;'
        'blood_nist,bone_nist,omnipaque_350_blood_nist',
    )
    parser.add_argument(
        '--vnc_maps',
        default='omnipaque_350_blood_nist:blood_nist'
    )
    parser.add_argument('--mono_energies', type=int, nargs='+', default=[70, 140])
    parser.add_argument('--valid_channels', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--slice_average', type=int, default=1)
    parser.add_argument('--display_window', type=float, nargs=2, default=[-160, 240])

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    args = utils.get_run_info(args)

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def load_calibration(filename, mat_names) -> np.ndarray:
    '''
    Load calibration coefficients for basis materials

    Parameters
    ----------
    filename : str
        Path to the CSV file containing calibration coefficients
    mat_names : list of str
        List of material names to extract

    Returns
    -------
    basis_mat : np.ndarray
        Array of shape (3, len(mat_names)) containing calibration coefficients
    '''

    manifest = pd.read_csv(filename)

    basis_mat = []
    for mat in mat_names:
        df = manifest[manifest['Material'] == mat]
        basis_mat.append([df[f'Channel{i}'].values[0] for i in range(3)])
    basis_mat = np.array(basis_mat).T

    return basis_mat


def load_mono_att(filename, mat_names, mono_energies) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load monochromatic attenuation coefficients for basis materials and water

    Parameters
    ----------
    filename : str
        Path to the CSV file containing attenuation coefficients
    mat_names : list of str
        List of material names to extract
    mono_energies : list of int
        List of monochromatic energies to extract

    Returns
    -------
    mono_att : np.ndarray
        Array of shape (len(mono_energies), len(mat_names)) containing attenuation coefficients
    water_att : np.ndarray
        Array of shape (len(mono_energies),) containing water attenuation coefficients
    '''
    manifest = pd.read_csv(filename)

    mono_att = []
    for mat in mat_names:
        att = []
        for energy in mono_energies:
            df = manifest[manifest['energy'] == energy]
            att.append(df[mat].values[0])
        mono_att.append(att)
    mono_att = np.array(mono_att).T

    water_att = []
    for energy in mono_energies:
        df = manifest[manifest['energy'] == energy]
        water_att.append(df['true_water'].values[0])
    water_att = np.array(water_att)

    return mono_att, water_att


def load_mmd_triplets(filename, mmd_triplets, energies) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load multi-material decomposition triplets

    Parameters
    ----------
    filename : str
        Path to the CSV file containing attenuation coefficients
    mmd_triplets : list of str
        List of triplet strings, each containing three material names separated by commas
    energies : list of int
        List of monochromatic energies to extract

    Returns
    -------
    mmd_mats : np.ndarray
        Array of shape (len(energies), number of unique materials) containing attenuation coefficients
    mmd_indices : np.ndarray
        Array of shape (number of triplets, 3) containing indices of materials in each triplet
    mmd_names : np.ndarray
        Array of unique material names
    '''
    mmd_names = np.unique(mmd_triplets)

    manifest = pd.read_csv(filename)
    manifest = manifest[['energy'] + list(mmd_names)]
    manifest_mono = []
    for energy in energies:
        df = manifest[manifest['energy'] == energy]
        manifest_mono.append(df)
    manifest_mono = pd.concat(manifest_mono)

    mmd_mats = []
    for name in mmd_names:
        mmd_mats.append(manifest_mono[name].values)
    mmd_mats = np.array(mmd_mats).T

    mmd_indices = []
    for triplet in mmd_triplets:
        mmd_indices.append([np.where(mmd_names == name)[0] for name in triplet])
    mmd_indices = np.array(mmd_indices)[..., 0]

    return mmd_mats, mmd_indices, mmd_names


def load_vnc_map(filename, vnc_maps, energies):
    manifest = pd.read_csv(filename)
    manifest_mono = []
    for energy in energies:
        df = manifest[manifest['energy'] == energy]
        manifest_mono.append(df)
    manifest_mono = pd.concat(manifest_mono)

    vnc_mono_map = {}
    for v in vnc_maps:
        vnc_mono_map[v[0]] = manifest_mono[v[1]].values

    return vnc_mono_map


# %%
def plot_mmd_triplets(mmd_mats, mmd_indices):
    '''
    Plot the MMD triplets in the two mono energies
    '''
    plt.figure()
    for triplet in mmd_indices:
        mats = mmd_mats[:, triplet]
        x = list(mats[0]) + [mats[0, 0]]
        y = list(mats[1]) + [mats[1, 0]]
        plt.plot(x, y, 'o-')
        plt.fill(x, y)
    plt.show()


# %%
def two_material_decomposition(imgs, basis_mats, channels=None):
    '''
    Two material decomposition is for iodine quantification
    '''
    if channels is not None:
        basis_mats = basis_mats[channels]
        imgs = imgs[channels]

    decomp_mat = basis_mats
    print('Two-mat Condition Number = ', np.linalg.cond(decomp_mat))

    y = imgs.reshape([imgs.shape[0], -1])

    x, _, _, _ = np.linalg.lstsq(decomp_mat, y, rcond=None)

    x = x.reshape([basis_mats.shape[1]] + list(imgs.shape[1:]))

    return x


def compose_mono_imgs(base_imgs, mono_mat):
    mono_imgs = []
    for energy in range(mono_mat.shape[1]):
        img = np.sum(base_imgs.transpose([1, 2, 3, 0]) * mono_mat[energy], -1)
        mono_imgs.append(img[np.newaxis])
    mono_imgs = np.concatenate(mono_imgs, 0)

    return mono_imgs


def three_material_decomposition(imgs, basis_mats, channels=None):
    '''
    Three material decomposition.
    '''
    if channels is not None:
        basis_mats = basis_mats[channels]
        imgs = imgs[channels]

    decomp_mat = np.concatenate([basis_mats, np.ones([1, basis_mats.shape[1]])], axis=0)
    print('Three-mat Condition Number = ', np.linalg.cond(decomp_mat))

    y = imgs.reshape([imgs.shape[0], -1])
    y = np.concatenate([y, np.ones([1, y.shape[1]])], axis=0)

    x, _, _, _ = np.linalg.lstsq(decomp_mat, y, rcond=None)

    x = x.reshape([basis_mats.shape[1]] + list(imgs.shape[1:]))

    return x


def multi_material_decomposition(imgs, mmd_mats, mmd_indices, channels=None):
    '''
    Multi-material decomposition using triplets
    '''
    if channels is not None:
        mmd_mats = mmd_mats[channels]
        imgs = imgs[channels]

    # first do three material decomposition for each triplet
    decomp_triplets = []
    for triplet in mmd_indices:
        mats = mmd_mats[:, triplet]
        decomp = three_material_decomposition(imgs, mats)
        decomp_triplets.append(decomp)
    decomp_triplets = np.array(decomp_triplets)

    # find which triplet each pixel is sitting in
    # it should be all positive. Since the triplets are non-overlapping,
    # one could look for the smallest decomp factors within each triplet,
    # then take the largest among all the triplets
    min_decomp_fac = np.min(decomp_triplets, 1)
    decomp_triplet_ind = np.argmax(min_decomp_fac, 0)
    # the decomp_triplet_fac can be used to determine out-of-triplet points and process later
    # decomp_triplet_fac = np.max(min_decomp_fac, 0)

    # assign all the basis mat images
    decomp_imgs = np.zeros([len(np.unique(mmd_indices))] + list(imgs.shape[1:]), np.float32)
    for itri, triplet in enumerate(mmd_indices):
        mask = np.where(decomp_triplet_ind == itri, 1, 0)
        for k, ind in enumerate(triplet):
            decomp_imgs[ind] += decomp_triplets[itri, k, ...] * mask

    return decomp_imgs


# %%
def get_side(pts, v1, v2):
    '''
    Find which side of the line (v1, v2) the points are on
    '''
    return (pts[:, 0] - v1[0]) * (v2[1] - v1[1]) - (v2[0] - v1[0]) * (pts[:, 1] - v1[1])


def is_inside_triangle(pts, v1, v2, v3):
    '''
    Check if the points are inside the triangle (v1, v2, v3)
    '''
    s1 = get_side(pts, v1, v2)
    s2 = get_side(pts, v2, v3)
    s3 = get_side(pts, v3, v1)

    all_pos = (s1 > 0) & (s2 > 0) & (s3 > 0)
    all_neg = (s1 < 0) & (s2 < 0) & (s3 < 0)
    return all_pos | all_neg


def proj_to_edge(pts, v1, v2, is_inside_tri: np.array):
    '''
    Project the points to the edge (v1, v2) of the triangle. If the points are inside the triangle,
    then the projection is the points themselves. Returns the projected points and the distance.
    '''
    t = (pts[:, 0] - v1[0]) * (v2[0] - v1[0]) + (pts[:, 1] - v1[1]) * (v2[1] - v1[1])
    t = t / ((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)

    t = np.clip(t, 0, 1)

    proj_pts = np.array([v1[0] + t * (v2[0] - v1[0]), v1[1] + t * (v2[1] - v1[1])]).T
    distance = np.sqrt((proj_pts[:, 0] - pts[:, 0]) ** 2 + (proj_pts[:, 1] - pts[:, 1]) ** 2)

    proj_pts[is_inside_tri] = pts[is_inside_tri]
    distance[is_inside_tri] = 0

    return proj_pts, distance


def proj_to_triplet(pts, v1, v2, v3):
    '''
    Project all the pts to the triangle (v1, v2, v3)
    '''
    is_inside_tri = is_inside_triangle(pts, v1, v2, v3)

    # project to each egde
    p1, d1 = proj_to_edge(pts, v1, v2, is_inside_tri)
    p2, d2 = proj_to_edge(pts, v2, v3, is_inside_tri)
    p3, d3 = proj_to_edge(pts, v3, v1, is_inside_tri)

    # select the closest point
    proj_pts = np.stack([p1, p2, p3], axis=-1)
    distance = np.stack([d1, d2, d3], axis=-1)
    idx = np.argmin(distance, axis=-1)

    proj_pts = proj_pts[np.arange(len(proj_pts)), :, idx]
    distance = distance[np.arange(len(distance)), idx]

    return proj_pts, distance


def proj_img_to_triplet(imgs, basis_mats, channels=None):
    '''
    Project each pixel to its closest triplet first. So that the previous three material decomposition
    function can be directly applied.
    '''
    if channels is not None:
        basis_mats = basis_mats[channels]
        imgs = imgs[channels]

    print('Projecting to triplets...', flush=True)
    pts = imgs.reshape([imgs.shape[0], -1]).T
    proj_pts, distance = proj_to_triplet(pts, basis_mats[:, 0], basis_mats[:, 1], basis_mats[:, 2])

    proj_imgs = proj_pts.T.reshape(imgs.shape)
    distance = distance.T.reshape(imgs.shape[1:])

    return proj_imgs, distance


def multi_material_decomposition_proj_to_triplet(imgs, mmd_mats, mmd_indices, channels=None):
    '''
    Project each pixel to its closest triplet first, then do three material decomposition
    '''
    if channels is not None:
        mmd_mats = mmd_mats[channels]
        imgs = imgs[channels]

    # first do three material decomposition for each triplet
    decomp_triplets = []
    decomp_triplets_dist = []
    for triplet in mmd_indices:
        mats = mmd_mats[:, triplet]
        proj_imgs, proj_dist = proj_img_to_triplet(imgs, mats)
        decomp = three_material_decomposition(proj_imgs, mats)
        decomp_triplets.append(decomp)
        decomp_triplets_dist.append(proj_dist)
    decomp_triplets = np.array(decomp_triplets)
    decomp_triplets_dist = np.array(decomp_triplets_dist)

    # determine which triplet each pixel is sitting in by the smallest distance
    decomp_triplet_ind = np.argmin(decomp_triplets_dist, 0)

    # assign all the basis mat images
    decomp_imgs = np.zeros([len(np.unique(mmd_indices))] + list(imgs.shape[1:]), np.float32)
    for itri, triplet in enumerate(mmd_indices):
        mask = np.where(decomp_triplet_ind == itri, 1, 0)
        for k, ind in enumerate(triplet):
            decomp_imgs[ind] += decomp_triplets[itri, k, ...] * mask

    return decomp_imgs


# %%
def display_results(imgs, windows, nrow, ncol, figsize, islice=None):
    plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        plt.subplot(nrow, ncol, i + 1)
        if islice is None:
            display_slice = imgs[i].shape[0] // 2
        else:
            display_slice = islice
        plt.imshow(imgs[i][display_slice], 'gray', vmin=windows[i][0], vmax=windows[i][1])
    plt.show()


# %%
def save_nii(img, filename, norm=1000, offset=-1000, dtype=np.int16, sitk_template=None, slice_average=1):
    img = (img * norm + offset).astype(dtype)
    sitk_img = sitk.GetImageFromArray(img)

    if sitk_template is not None:
        if slice_average == 1:
            sitk_img.CopyInformation(sitk_template)
        else:
            sitk_img.SetOrigin(sitk_template.GetOrigin())
            sitk_img.SetDirection(sitk_template.GetDirection())
            spacing = sitk_template.GetSpacing()
            sitk_img.SetSpacing([spacing[0], spacing[1], spacing[2] * slice_average])

    sitk.WriteImage(sitk_img, filename)


# %%
def main(args):
    input_scan_dir = os.path.join(base_input_dir, args.input_scan)
    input_calib_filename = os.path.join(base_input_dir, args.input_calib)
    input_mono_att_filename = os.path.join(base_input_dir, args.input_mono_att)
    output_dir = os.path.join(base_output_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # normalize display window
    display_window = [(v + 1000) / 1000 for v in args.display_window]

    # compose mmd triplets
    mmd_triplets = [s.split(',') for s in args.mmd_triplets.split(';')]
    vnc_maps = [s.split(':') for s in args.vnc_maps.split(';')]

    print('Loading Decomposition Coefficients...', flush=True)
    base_names = [args.base_mat_0, args.base_mat_1]
    base_mat = load_calibration(input_calib_filename, base_names)
    base_mat = (base_mat + 1000) / 1000  # normalize
    mono_mat, water_att = load_mono_att(input_mono_att_filename, base_names, args.mono_energies)
    mmd_mats, mmd_indices, mmd_names = load_mmd_triplets(input_mono_att_filename, mmd_triplets, args.mono_energies)
    vnc_maps = load_vnc_map(input_mono_att_filename, vnc_maps, args.mono_energies)

    if args.debug:
        plot_mmd_triplets(mmd_mats, mmd_indices)

    print('Loading Images...', flush=True)
    imgs = utils.load_image(input_scan_dir)[0]
    imgs = (imgs + 1000) / 1000  # normalize
    if args.slice_average > 1:
        nz_trunc = imgs.shape[1] // args.slice_average * args.slice_average
        imgs = imgs[:, :nz_trunc]
        imgs = imgs.reshape([imgs.shape[0], -1, args.slice_average, imgs.shape[2], imgs.shape[3]]).mean(2)
    if args.debug:
        display_results(
            [imgs[0], imgs[1], imgs[2]],
            [display_window] * 3,
            1, 3, (15, 5)
        )

    print('Two material decomposition', flush=True)
    img_basis = two_material_decomposition(imgs, base_mat, args.valid_channels)
    if args.debug:
        display_results(
            [img_basis[0], img_basis[1]],
            [(0, 1)] * 2,
            1, 2, (10, 5)
        )

    print('Get monochromatic images', flush=True)
    img_mono = compose_mono_imgs(img_basis, mono_mat)
    if args.debug:
        display_results(
            [img_mono[0] / water_att[0], img_mono[1] / water_att[1]],
            [display_window] * 2,
            1, 2, (10, 5)
        )

    print('Multi material decomposition')
    print(mmd_names, flush=True)
    print(mmd_mats, flush=True)
    # img_mmd = multi_material_decomposition(img_mono, mmd_mats, mmd_indices)
    img_mmd = multi_material_decomposition_proj_to_triplet(img_mono, mmd_mats, mmd_indices)
    if args.debug:
        display_results(
            [img_mmd[k] for k in range(img_mmd.shape[0])],
            [(0, 1)] * img_mmd.shape[0],
            2, 3, (15, 10),
        )

    print('Get VNC images')
    vnc_mats = mmd_mats.copy()
    for i in range(vnc_mats.shape[1]):
        mat_name = mmd_names[i]
        if mat_name in vnc_maps:
            vnc_mats[:, i] = vnc_maps[mat_name]
    print(vnc_mats, flush=True)

    img_vnc = []
    for ienergy in range(mmd_mats.shape[0]):
        img = np.sum(img_mmd.transpose([1, 2, 3, 0]) * vnc_mats[ienergy], -1)
        img_vnc.append(img)
    img_vnc = np.array(img_vnc)
    if args.debug:
        display_results(
            [img_vnc[0] / water_att[0], img_vnc[1] / water_att[1]],
            [display_window] * 2,
            1, 2, (10, 5)
        )

    # save the results
    # template nii
    sitk_template = sitk.ReadImage(os.path.join(input_scan_dir, 'img_ch0.nii.gz'))
    # save basis images
    print('Saving basis...', flush=True)
    for i in range(img_basis.shape[0]):
        save_nii(
            img_basis[i],
            os.path.join(output_dir, 'base_{0}.nii.gz'.format(base_names[i])),
            offset=0,
            sitk_template=sitk_template,
            slice_average=args.slice_average
        )
    # save mono images
    print('Saving mono...', flush=True)
    for i in range(img_mono.shape[0]):
        save_nii(
            img_mono[i] / water_att[i],
            os.path.join(output_dir, 'mono_{0}_keV.nii.gz'.format(args.mono_energies[i])),
            sitk_template=sitk_template,
            slice_average=args.slice_average
        )
    # save MMD images
    print('Saving MMD...', flush=True)
    for i in range(img_mmd.shape[0]):
        save_nii(
            img_mmd[i],
            os.path.join(output_dir, 'mmd_{0}.nii.gz'.format(mmd_names[i])),
            offset=0,
            sitk_template=sitk_template,
            slice_average=args.slice_average
        )
    # save VNC images
    print('Saving vnc...', flush=True)
    for i in range(img_vnc.shape[0]):
        save_nii(
            img_vnc[i] / water_att[i],
            os.path.join(output_dir, 'vnc_{0}_keV.nii.gz'.format(args.mono_energies[i])),
            sitk_template=sitk_template,
            slice_average=args.slice_average
        )
    print('All done')

    return base_mat, mono_mat, base_names, mmd_mats, mmd_indices, mmd_names


# %%
if __name__ == '__main__':
    args = get_args([
        # '--input_scan', 'omnitom_pcd/calibration/calibration_42/img/0',
        # '--output_dir', 'md_baseline/calibration/calibration_42/img/0/img_mmd',
        # '--input_calib', 'omnitom_pcd/calibration/calibration_42/img_md_calib_avg.csv',
        # '--mmd_triplets', 'air,blood_100,iodine_5;blood_100,calcium_100,iodine_5',
        # '--vnc_maps', 'iodine_5:blood_100',

        '--input_scan', 'omnitom_pcd/recon/img/3',
        '--output_dir', 'md_baseline/recon/img/3/img_mmd',
        '--input_calib', 'omnitom_pcd/calibration/calibration_1_7/img_md_calib_avg.csv',
        '--slice_average', '6',
        '--display_window', '0', '100',
    ])

    res = main(args)

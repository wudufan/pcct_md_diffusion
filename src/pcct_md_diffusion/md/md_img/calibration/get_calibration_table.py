'''
Calculate the averaged HU value for each material in the calibration phantom
'''

# %%
import argparse
import os
import sys
import subprocess
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd

from typing import List

import pcct_md_diffusion.utils as utils
from pcct_md_diffusion.locations import base_input_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--subfolders', default=None)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    args.datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    args.user = os.getenv('USER')
    args.sys_argv = sys.argv
    args.script = os.path.abspath(__file__)

    for k, v in vars(args).items():
        print(f'{k} = {v}', flush=True)

    return args


# %%
def get_roi_mean(imgs: np.array, seg: np.array, manifest: pd.DataFrame):
    for i in range(imgs.shape[0]):
        manifest['Channel{0}'.format(i)] = None

    for i, row in manifest.iterrows():
        for k in range(len(imgs)):
            manifest.at[i, 'Channel{0}'.format(k)] = np.mean(imgs[k][seg == row['Seg']])

    return manifest


def get_avg_roi_mean(manifests: List[pd.DataFrame]):
    '''
    Average the values from the rows with the same 'Material' entries
    '''
    df_all = pd.concat(manifests, ignore_index=True)
    df_avg = df_all.groupby('Material').mean().reset_index()
    df_avg = df_avg.drop(columns=['Seg'])

    return df_avg


# %%
def main(args):
    input_dir = os.path.join(base_input_dir, args.input_dir)
    output_dir = os.path.join(base_input_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.subfolders is not None:
        subfolders = args.subfolders.split(',')
    else:
        subfolders = sorted([f.name for f in os.scandir(input_dir) if f.is_dir()])

    df_roi_means = []
    for subfolder in subfolders:
        print(f'Processing subfolder: {subfolder}', flush=True)
        img = utils.load_image(os.path.join(input_dir, subfolder))[0]
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, subfolder, 'segmentation.seg.nrrd')))
        df_materials = pd.read_csv(os.path.join(input_dir, subfolder, 'materials.csv'))

        df_roi_mean = get_roi_mean(img, seg, df_materials)

        df_roi_means.append(df_roi_mean)
        df_roi_mean.to_csv(os.path.join(output_dir, f'img_md_calib_{subfolder}.csv'), index=False)

    df_roi_means = get_avg_roi_mean(df_roi_means)
    df_roi_means.to_csv(os.path.join(output_dir, 'img_md_calib_avg.csv'), index=False)

    return df_roi_means


# %%
if __name__ == '__main__':
    args = get_args([
        '--input_dir', 'omnitom_pcd/calibration/calibration_1_7/img',
        '--output_dir', 'omnitom_pcd/calibration/calibration_1_7',
        # '--subfolders', '0',
    ])

    res = main(args)

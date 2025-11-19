'''
Utility functions
'''

# %%
import configparser
import os
import glob

import numpy as np
import SimpleITK as sitk

from typing import List, Tuple, Union
from enum import Enum
from leapctype import tomographicModels


# %%
class GeometryType(Enum):
    PARALLEL = 'parallel'
    FAN_FLAT = 'fan_flat'
    CONE_CURVED = 'cone_curved'
    CONE_FLAT = 'cone_flat'


class StandardGeometry:
    """Geometry class to hold geometry parameters."""

    def __init__(self):
        self.nview = int(1440)
        self.rotview = int(1440)
        self.nu = int(512)
        self.nv = int(1)
        self.nx = int(512)
        self.ny = int(512)
        self.nz = int(1)
        self.dx = np.float32(1.0)
        self.dy = np.float32(1.0)
        self.dz = np.float32(1.0)
        self.cx = np.float32(0.0)
        self.cy = np.float32(0.0)
        self.cz = np.float32(0.0)
        self.dsd = np.float32(1085.6)
        self.dso = np.float32(595)
        self.du = np.float32(1.0)
        self.dv = np.float32(1.0)
        self.off_u = np.float32(0.0)
        self.off_v = np.float32(0.0)
        self.helical_pitch = np.float32(0.0)

    def __repr__(self):
        attrs = vars(self)
        return 'StandardGeometry:\n' + '\n'.join(f'{key}: {value}' for key, value in attrs.items())


def load_geometry(filename: str) -> StandardGeometry:
    """Load geometry from config file.

    Args:
        filename (str): Config file name without suffix.
    Returns:
        StandardGeometry: Loaded geometry.
    """
    geo = StandardGeometry()
    config = configparser.ConfigParser()
    config.read(filename)

    for key, value in config['geometry'].items():
        if key.endswith('_mm'):
            key = key[:-3]
        if hasattr(geo, key):
            attr_type = type(getattr(geo, key))
            setattr(geo, key, attr_type(value))

    return geo


def set_leapct_geometry(
    geo: StandardGeometry,
    geometry_type: GeometryType = GeometryType.PARALLEL,
    angles_in_deg: np.ndarray = None,
) -> tomographicModels:
    leapct = tomographicModels()

    if angles_in_deg is None:
        angles_in_deg = leapct.setAngleArray(geo.nview, 360.0)

    if geometry_type == GeometryType.PARALLEL:
        leapct.set_parallelbeam(
            numAngles=geo.nview,
            numRows=geo.nv,
            numCols=geo.nu,
            pixelHeight=geo.dv,
            pixelWidth=geo.du,
            centerRow=(geo.nv - 1) / 2.0 + geo.off_v,
            centerCol=(geo.nu - 1) / 2.0 + geo.off_u,
            phis=angles_in_deg,
        )
    elif geometry_type == GeometryType.FAN_FLAT:
        leapct.set_fanbeam(
            numAngles=geo.nview,
            numRows=geo.nv,
            numCols=geo.nu,
            pixelHeight=geo.dv,
            pixelWidth=geo.du,
            centerRow=(geo.nv - 1) / 2.0 + geo.off_v,
            centerCol=(geo.nu - 1) / 2.0 + geo.off_u,
            sod=geo.dso,
            sdd=geo.dsd,
            phis=angles_in_deg,
        )
    elif geometry_type == GeometryType.CONE_CURVED or geometry_type == GeometryType.CONE_FLAT:
        leapct.set_conebeam(
            numAngles=geo.nview,
            numRows=geo.nv,
            numCols=geo.nu,
            pixelHeight=geo.dv,
            pixelWidth=geo.du,
            centerRow=(geo.nv - 1) / 2.0 + geo.off_v,
            centerCol=(geo.nu - 1) / 2.0 + geo.off_u,
            sod=geo.dso,
            sdd=geo.dsd,
            helicalPitch=geo.helical_pitch,
            phis=angles_in_deg
        )
        if geometry_type == GeometryType.CONE_CURVED:
            leapct.set_curvedDetector()
        else:
            leapct.set_flatDetector()
    else:
        raise NotImplementedError(f'Geometry type {geometry_type} not implemented.')

    leapct.set_volume(
        numX=geo.nx,
        numY=geo.ny,
        numZ=geo.nz,
        voxelWidth=geo.dx,
        voxelHeight=geo.dz,
        offsetX=geo.cx,
        offsetY=geo.cy,
        offsetZ=geo.cz,
    )

    return leapct


# %%
def load_sinogram(
        foldername: str,
        channel: Union[List[int], int] = None,
        num_rotations_to_average: int = 2,
        prefix='prj_ch',
        first_rotation: int = 0,
        last_rotation: int = -1,
) -> Tuple[sitk.Image, float, float]:
    """Load sinogram from folder.

    Args:
        foldername (str): Folder name containing sinogram files.
        channel (List[int]): List of channel indices to load.
        num_rotations_to_average (int): Number of rotations to average.
        prefix (str): Prefix of sinogram file names.
        first_rotation (int): First rotation index to load.
        last_rotation (int): Last rotation index to load.
    Returns:
        sitk.Image: Loaded sinogram image, shape (nchannel, nview, nv*number_of_rotations, nu).
        float: Pixel size in u direction.
        float: Pixel size in v direction.
    """
    # in case channel is int
    if isinstance(channel, int):
        channel = [channel]
    elif channel is None:
        # find all available channels
        filenames = glob.glob(os.path.join(foldername, f'{prefix}*_rot*.nii.gz'))
        channel = sorted(list(set(int(os.path.basename(fname).split('_')[1][2]) for fname in filenames)))

    all_sinograms = []
    for ch in channel:
        print(f'Loading channel {ch}...', flush=True)
        filenames = glob.glob(os.path.join(foldername, f'{prefix}{ch}_rot*.nii.gz'))
        filenames.sort()

        if last_rotation < 0:
            last_rotation = len(filenames) - 1

        filenames = filenames[first_rotation:last_rotation + 1]
        sinograms_ch = []
        print(f'Loading rotations ({len(filenames)}):', flush=True, end=' ')
        for i, fname in enumerate(filenames):
            print(f'{i} ', end='', flush=True)
            sino = sitk.ReadImage(fname)
            sinograms_ch.append(sino)
        print('done.', flush=True)

        du = sinograms_ch[0].GetSpacing()[0]
        dv = sinograms_ch[0].GetSpacing()[1]

        # remove the sinogram that has a different dimension compared to the first sinogram
        sinograms_ch = [sino for sino in sinograms_ch if sino.GetSize() == sinograms_ch[0].GetSize()]
        print(f'After removing inconsistent sinograms, {len(sinograms_ch)} sinograms remain.', flush=True)

        sinograms_ch = np.array([sitk.GetArrayFromImage(sino) for sino in sinograms_ch])
        # average every 'num_rotations_to_average' rotations
        if num_rotations_to_average > 1:
            sinograms_ch = sinograms_ch.reshape(-1, num_rotations_to_average, *sinograms_ch.shape[1:]).mean(axis=1)

        # join alone v direction
        sinograms_ch = np.concatenate(sinograms_ch, axis=1)
        all_sinograms.append(sinograms_ch)

    # stack along channel dimension
    all_sinograms = np.array(all_sinograms).astype(np.float32)

    return all_sinograms, du, dv


def average_projection_slices(sinogram, dv, num_slices_to_average=2) -> Tuple[np.ndarray, float]:
    """Average projection slices in the v direction.

    Args:
        sinogram (np.ndarray): Input sinogram of shape (nchannel, nview, nv, nu).
        dv (float): Original pixel size in v direction.
        num_slices_to_average (int): Number of slices to average.
    Returns:
        np.ndarray: Averaged sinogram.
        float: New pixel size in v direction.
    """
    sinogram = sinogram.reshape(
        sinogram.shape[0],
        sinogram.shape[1],
        sinogram.shape[2] // num_slices_to_average,
        num_slices_to_average,
        sinogram.shape[3],
    ).mean(axis=3)

    new_dv = dv * num_slices_to_average
    return sinogram, new_dv


# %%
def load_image(
        foldername: str,
        channel: Union[List[int], int] = None,
        prefix='img_ch',
) -> Tuple[np.array, float, float, float]:
    """Load image from folder.

    Args:
        foldername (str): Folder name containing image files.
        channel (int): Channel index to load.
    Returns:
        np.array: Loaded image array, shape (nchannel, nz, ny, nx).
        float: Voxel size in x direction.
        float: Voxel size in y direction.
        float: Voxel size in z direction.
    """
    # in case channel is int
    if isinstance(channel, int):
        channel = [channel]
    elif channel is None:
        # find all available channels
        filenames = glob.glob(os.path.join(foldername, f'{prefix}*.nii.gz'))
        channel = sorted(list(set(int(os.path.basename(fname).split('_')[1][2]) for fname in filenames)))

    all_images = []
    for ch in channel:
        print(f'Loading channel {ch}...', flush=True)
        sitk_img = sitk.ReadImage(os.path.join(foldername, f'{prefix}{ch}.nii.gz'))
        img = sitk.GetArrayFromImage(sitk_img)
        all_images.append(img)

        dx, dy, dz = sitk_img.GetSpacing()

    all_images = np.array(all_images).astype(np.float32)
    return all_images, dx, dy, dz


# %%
def add_noise_gaussian(prj: np.ndarray, N0: float, dose_factor: float, seed: int = None) -> np.ndarray:
    '''Add Gaussian noise to the projection data to simulate a lower dose acquisition.

    Args:
        prj (np.ndarray): Input projection data.
        N0 (float): Incident photon number.
        dose_factor (float): Dose reduction factor (0 < dose_factor <= 1).
        seed (int): Random seed for reproducibility.
    Returns:
        np.ndarray: Noisy projection data.
    '''
    if seed is not None:
        np.random.seed(seed)

    # add noise
    if N0 > 0 and dose_factor < 1:
        prj = prj + np.sqrt((1 - dose_factor) / dose_factor * np.exp(prj) / N0) * np.random.normal(size=prj.shape)
        prj = prj.astype(np.float32)

    return prj


# %%
if __name__ == '__main__':
    from pcct_md_diffusion.locations import base_input_dir

    geo = load_geometry(os.path.join(base_input_dir, 'omnitom_pcd/geometry/pcd_parallel_3x3_512.cfg'))

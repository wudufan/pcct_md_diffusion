'''
Load attenuation data from NIST files.
'''

# %%
import os
import pandas as pd
import numpy as np


# %%
def load_nist_attenuation(
    filename: str,
    density: float = None,
    kevs: np.array = np.arange(20, 120 + 1, 1),
) -> pd.DataFrame:
    """Load NIST attenuation data from file.

    Args:
        filename (str): File name of the attenuation data.
        density (float): Density of the material in g/cm^3 (optional). Read from the file if not provided.
        kevs (np.array): Array of energy values in keV.
    Returns:
        pd.DataFrame: Loaded normalized attenuation data with columns 'Energy' in keV and 'mu' in mm^-1.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    if density is None:
        for line in lines:
            if line.startswith('density'):
                density = float(line.split('=')[1].strip())
                break
    assert density is not None, "Density not found in the file. Please provide it as an argument."

    energies = []
    mus = []
    for line in lines:
        tokens = [t for t in line.split() if t != '']
        if len(tokens) == 3 or len(tokens) == 4:
            energies.append(float(tokens[-3]) * 1000)  # convert from MeV to keV
            mus.append(float(tokens[-2]) * density / 10)  # convert from cm^2/g to mm^-1
    energies = np.array(energies)
    mus = np.array(mus)

    # resample in the log-log domain
    log_energies = np.log(energies)
    log_mus = np.log(mus)
    energies_resampled = kevs
    log_mus_resampled = np.interp(np.log(energies_resampled), log_energies, log_mus)
    mus_resampled = np.exp(log_mus_resampled)

    df = pd.DataFrame({'energy': energies_resampled, 'mu': mus_resampled})
    return df


def load_mixture_attenuation(
    base_dir: str,
    components: list,
    mass_fractions: list,
    density: float,
    kevs: np.array = np.arange(20, 120 + 1, 1),
) -> pd.DataFrame:
    """Load attenuation data for a mixture of materials.

    Args:
        base_dir (str): Base directory where the NIST files are located.
        components (list): List of component names (e.g., ['H', 'Na']).
        mass_fractions (list): List of mass fractions for each component.
        density (float): Density of the mixture in g/cm^3.
        kevs (np.array): Array of energy values in keV.
    Returns:
        pd.DataFrame: Loaded normalized attenuation data with columns 'Energy' in keV and 'mu' in mm^-1.
    """
    assert len(components) == len(mass_fractions), "Components and mass_fractions must have the same length."

    mass_fractions = np.array(mass_fractions)
    mass_fractions = mass_fractions / mass_fractions.sum()  # normalize mass fractions

    mass_mus = []
    for comp, mf in zip(components, mass_fractions):
        df_comp = load_nist_attenuation(
            filename=os.path.join(base_dir, f'{comp}.txt'),
            density=1,  # use mass attenuation
            kevs=kevs
        )
        mass_mus.append(df_comp['mu'].values * mf * 10)  # in cm^2/g
    mass_mus = np.array(mass_mus)
    total_mass_mu = np.sum(mass_mus, axis=0)  # in cm^2/g
    total_mu = total_mass_mu * density / 10  # convert to mm^-1

    df = pd.DataFrame({'energy': kevs, 'mu': total_mu})

    return df


def load_mixture_attenuation_from_composition_file(
    composition_file: str,
    attenuation_file_dir: str,
    kevs: np.array = np.arange(20, 120 + 1, 1),
    return_density: bool = False
) -> pd.DataFrame:
    """Load attenuation data for a mixture of materials from a composition file.

    Args:
        composition_file (str): Path to the composition file (CSV) with columns
            'composition', 'density', and mass fraction for each element
        attenuation_file_dir (str): Base directory where the NIST files are located.
        kevs (np.array): Array of energy values in keV.
    Returns:
        pd.DataFrame: Loaded normalized attenuation data with columns 'Energy' in keV and 'mu' in mm^-1.
    """
    df_comp = pd.read_csv(composition_file, dtype=str)
    compositions = df_comp['composition'].values
    mus = []

    elements = [c for c in df_comp.columns if c not in ['composition', 'density', 'FullName', 'rho_e']]
    for elem in elements:
        df_comp[elem] = df_comp[elem].str.rstrip('%').astype(float) / 100
    df_comp['density'] = df_comp['density'].astype(float)
    df_comp['rho_e'] = df_comp['rho_e'].astype(float)

    for i, row in df_comp.iterrows():
        density = row['density']
        mass_fractions = row[elements].values
        df_mixture = load_mixture_attenuation(
            base_dir=attenuation_file_dir,
            components=elements,
            mass_fractions=mass_fractions,
            density=density,
            kevs=kevs
        )
        mus.append(df_mixture['mu'].values)

    df_res = pd.DataFrame({'energy': kevs})
    for comp, mu in zip(compositions, mus):
        df_res[comp] = mu

    if return_density:
        dict_density = {row['composition']: row['density'] for _, row in df_comp.iterrows()}

        return df_res, dict_density

    return df_res


# %%
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    from pcct_md_diffusion.locations import base_input_dir

    df_mect_phantom = load_mixture_attenuation_from_composition_file(
        composition_file=os.path.join(base_input_dir, 'omnitom_pcd/calibration/material_composition.csv'),
        attenuation_file_dir=os.path.join(base_input_dir, 'spectrum/NIST')
    )

    # df = load_nist_attenuation(
    #     os.path.join(base_input_dir, 'spectrum/NIST/Gd.txt'),
    #     density=1  # use mass attenuation
    # )

    # plt.plot(df['Energy'], df['mu'])
    # plt.xlabel('Energy (keV)')
    # plt.ylabel('Attenuation Coefficient (mm^-1)')
    # plt.grid()
    # plt.show()

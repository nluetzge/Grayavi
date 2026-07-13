# IDENT			data_utils.py
# LANGUAGE		Python
# AUTHOR		N. LUETZGENDORF
# PURPOSE	    Utility functions for grayavi.py for anything related to data handling.
#
# VERSION
# 1.0.0 12.08.2025 NL Creation
# ===============================================================
# Imports
# ===============================================================
import numpy as np
import quaternionic
import spherical
import h5py
import yaml

from math_utils import activation, deactivation, compute_spin_vector


# ===============================================================
# Functions
# ===============================================================
class ConfigParam:
    """
    Represents a single configuration parameter, including metadata for automatic filename building.

    Attributes:
        value: The actual value of the parameter.
        flag (bool): Whether this parameter should appear in the generated filename.
        short (str): Short label to use in the filename (e.g., 'sr', 'fr').
        fmt (str): Python format string for the value in the filename.
    """

    def __init__(self, value, flag=False, short="", fmt="{}"):
        self.value = value
        self.flag = flag
        self.short = short
        self.fmt = fmt


class Config:
    """
    Loads a YAML configuration file with metadata, allows dict-style access to parameter values,
    and supports automatic filename generation from flagged parameters.

    Example usage:
        config = Config("config.yaml")
        print(config["xdim"])            # returns the value of xdim
        config["xdim"] = 1920            # modify a parameter
        filename = config.build_filename()  # build a filename using flagged parameters
    """

    def __init__(self, yaml_file):
        """
        Initialize the Config object by reading a YAML file.

        Args:
            yaml_file (str): Path to the YAML configuration file.
        """
        self.params = {}
        self.load_yaml(yaml_file)

    def load_yaml(self, yaml_file):
        """
        Load YAML file and convert entries into ConfigParam objects.

        Args:
            yaml_file (str): Path to YAML configuration file.
        """
        with open(yaml_file, "r") as f:
            raw = yaml.safe_load(f)
        for key, val in raw.items():
            if isinstance(val, dict):
                self.params[key] = ConfigParam(
                    value=val.get("value"),
                    flag=val.get("flag", False),
                    short=val.get("short", ""),
                    fmt=val.get("fmt", "{}")
                )
            else:
                # fallback if simple value is provided
                self.params[key] = ConfigParam(value=val)


    def __getitem__(self, key):
        """
        Allow dictionary-style access to return the parameter's value directly.

        Args:
            key (str): Parameter name.

        Returns:
            The value of the parameter.
        """
        return self.params[key].value

    def __setitem__(self, key, value):
        """
        Allow modifying a parameter value via dictionary-style access.

        Args:
            key (str): Parameter name.
            value: New value for the parameter.
        """
        if key in self.params:
            self.params[key].value = value
        else:
            self.params[key] = ConfigParam(value)

    def build_filename(self):
        """
        Construct a filename string from all flagged parameters.

        Returns:
            str: Filename composed of all flagged parameters in the order they appear in the YAML.
        """
        parts = []
        for key, param in self.params.items():
            if param.flag:
                if param.short:
                    parts.append(f"{param.short}-{param.fmt.format(param.value)}")
                else:
                    parts.append(param.fmt.format(param.value))
        return "_".join(parts)


# Create a dummy waveform for testing
def dummy_waveform(n_frames):
    """
    Creates a dummy sinus waveform with real and imaginary part

    Parameters
    ----------
    n_frames : int
        Number of points in the waveform.

    Returns
    -------
    time_vector : numpy.array
        Time array of the waveform
    waveform_td : numpy.array
        Normalized complex waveform
    """
    dt = 0.1
    time_vector = np.arange(n_frames) * dt
    waveform_td = np.exp(1j * 20 * np.pi * time_vector / (np.max(time_vector)))  # simple chirp
    waveform_td /= np.max(np.abs(waveform_td))
    return time_vector, waveform_td


def read_waveform(waveform_filename, n_frames, ts=1, rs=1):
    """
    Reads in the waveform from a hdf5 file. File needs to be of the format of:

    time_vector, shape=(N,), dtype=float64
    traj_x, shape=(N,), dtype=float64
    traj_y, shape=(N,), dtype=float64
    traj_z, shape=(N,), dtype=float64
    waveform_td, shape=(2, N), dtype=float64

    Waveform has real and imaginary part, N and m can be anything but N should be larger than
    n_frames

    Parameters
    ----------
    waveform_filename : str
        Name of the waveform file
    n_frames : int
        Number of frames/points of the final waveform (it will just be cut to this number)
    ts : float
        Time scale (factor by which the time vector will be multiplied)
    rs : float
        Radial scale (factor by which the trajectory coordinates will be multiplied)

    Returns
    -------
    time_vector : numpy.array
        Time array of the waveform
    waveform_td : numpy.array
        Normalized complex waveform
    L : numpy.array
        The spin of the constellation as a function of time
    points : numpy.array
        3D points of trajectory of first object
    points2 : numpy.array
        3D points of trajectory of second object (none for EMRIs usually)

    """
    # Read the file
    with h5py.File(waveform_filename, "r") as f:
        time_vector = f['time_vector'][:] * ts
        waveform = f['waveform_td'][:]
        traj_x = f['traj_x'][:] * rs
        traj_y = f['traj_y'][:] * rs
        traj_z = f['traj_z'][:] * rs
        # Check if there is a second moving body
        try:
            traj_x2 = f['traj_x2'][:] * rs
            traj_y2 = f['traj_y2'][:] * rs
            traj_z2 = f['traj_z2'][:] * rs
        except KeyError:
            traj_x2 = traj_y2 = traj_z2 = None

    # Trim the vectors to fint n_frames
    time_vector = time_vector[:n_frames]
    waveform = waveform[:, :n_frames]
    traj_x = traj_x[:n_frames]
    traj_y = traj_y[:n_frames]
    traj_z = traj_z[:n_frames]
    # This code started out with EMRIs so thats why I first didn't consider the second object
    if traj_x2 is not None:
        traj_x2 = traj_x2[:n_frames]
        traj_y2 = traj_y2[:n_frames]
        traj_z2 = traj_z2[:n_frames]
        points2 = np.array([traj_x2, traj_y2, traj_z2])
    else:
        points2 = None

    points = np.array([traj_x, traj_y, traj_z])
    waveform /= np.max(np.abs(waveform))

    # Create the complex waveform
    waveform_td = waveform[0, :].flatten() + 1j * waveform[1, :].flatten()

    # Compute the spin
    L = compute_spin_vector(points, dt=time_vector[1] - time_vector[0])

    return time_vector, waveform_td, L, points, points2


def swsh_grid(l, m, th, phi, r, ell_max, spin_weight, size, activation_offset, activation_width,
              deactivation_width):
    """
    Generate a spin-weighted spherical harmonic (SWSH) grid with activation/deactivation screening.

    This function computes spin-weighted spherical harmonics on a given spherical grid, applies
    activation and deactivation screening functions, and combines the `(l, m)` and `(l, -m)` modes
    to produce a complete grid. The result is typically used in gravitational wave visualization.

    Parameters
    ----------
    l : int
        Spherical harmonic degree (ℓ).
    m : int
        Spherical harmonic order (m).
    th : ndarray of shape (N,N,N)
        Polar (colatitude) angles in radians.
    phi : ndarray of shape (N,N,N)
        Azimuthal angles in radians.
    r : ndarray of shape (N,N,N)
        Radial coordinates.
    ell_max : int
        Maximum spherical harmonic degree used in the Wigner function computation.
    spin_weight : int
        Spin weight of the spherical harmonic (e.g., `-2` for gravitational waves).
    size : float
        Characteristic outer size of the grid, used in the deactivation function.
    activation_offset : float
        Radial offset at which the activation function starts.
    activation_width : float
        Radial width over which the activation function ramps up.
    deactivation_width : float
        Radial width over which the deactivation function ramps down.

    Returns
    -------
    swsh_lm : ndarray
        Complex-valued array of the combined `(l, m)` and `(l, -m)` spin-weighted spherical
        harmonics after activation/deactivation screening. The shape depends on the input `th`,
        `phi`, and `r` grids and the `ell_max` parameter.

    Notes
    -----
    - The quaternionic rotation angles are computed from spherical coordinates `(th, phi)`.
    - The spin-weighted spherical harmonics are computed using
      `spherical.Wigner(ell_max).sYlm(s=spin_weight, R=angles)`.
    - The activation function attenuates values near the origin to create a hollow center,
      while the deactivation function attenuates the outer radial region to smoothly fade the
      grid edges.
    - Combination of `(l, m)` and `(l, -m)` modes is performed to avoid missing parts of the grid.
      The theoretical combination formula involving conjugates does not produce the desired
      pole asymmetry for visualization purposes.
    """
    # swsh grid Setup
    # Quaternionic rotation angles
    angles = quaternionic.array.from_spherical_coordinates(th, phi)

    # Get spin-weighted spherical harmonics
    swsh = spherical.Wigner(ell_max).sYlm(s=spin_weight, R=angles)

    # Apply screening
    screen = (activation(r - activation_offset, activation_width) *
              deactivation(r, deactivation_width, size))
    swsh *= screen.reshape(screen.shape + (1,))

    # Combining l,m and l,-m into one grid (if we don't do this we only have half a grid)
    # Theoretically I think one should do wave*swsh_lm + wave.conj()*swsh_l-m but this doesn't give
    # you a nice asymmetry around the poles...
    swsh_lm = swsh[:, :, :, l * (l + 1) + m] + swsh[:, :, :, l * (l + 1) - m]  # Indexing: l,
    # m+ell_max
    return swsh_lm

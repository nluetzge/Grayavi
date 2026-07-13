# IDENT			math_utils.py
# LANGUAGE		Python
# AUTHOR		N. LUETZGENDORF
# PURPOSE	    Math utility functions for grayavi.py.
#
# VERSION
# 1.0.0 12.08.2025 NL Creation
# ===============================================================
# Imports
# ===============================================================
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates
from scipy.interpolate import interp1d


def map_range(a, a1, a2, b1=0.0, b2=1.0):
    """
    Linearly map a value or array `a` from range [a1, a2] to [b1, b2].

    Parameters:
        a  : float or np.ndarray
            Input value(s) to be mapped.
        a1 : float
            Lower bound of input range.
        a2 : float
            Upper bound of input range.
        b1 : float, optional
            Lower bound of target range (default: 0.0).
        b2 : float, optional
            Upper bound of target range (default: 1.0).

    Returns:
        float or np.ndarray
            Mapped value(s) in the range [b1, b2].
    """
    return b1 + ((a - a1) / (a2 - a1)) * (b2 - b1)


def smoothstep(x):
    """
    Smoothstep function for smooth interpolation between 0 and 1.

    The function is:
        0                for x < 0
        3x² - 2x³        for 0 ≤ x ≤ 1
        1                for x > 1

    Parameters:
        x : float or np.ndarray
            Input value(s).

    Returns:
        float or np.ndarray
            Smoothstep output.
    """
    return np.where(x < 0, 0, np.where(x <= 1, 3 * x**2 - 2 * x**3, 1))


def activation(x, width):
    """
    Activation function that applies smoothstep scaling.

    Parameters:
        x     : float or np.ndarray
            Input value(s).
        width : float
            Width scaling factor for the transition.

    Returns:
        float or np.ndarray
            Activated value(s).
    """
    return smoothstep(x / width)


def deactivation(x, width, outer):
    """
    Deactivation function that reverses smoothstep scaling from a boundary.

    Parameters:
        x     : float or np.ndarray
            Input value(s).
        width : float
            Width scaling factor for the transition.
        outer : float
            Outer boundary value for scaling.

    Returns:
        float or np.ndarray
            Deactivated value(s).
    """
    return smoothstep((outer - x) / width)


def normalize(v):
    """
    Normalize a vector to unit length.

    Parameters:
        v : np.ndarray, shape (n,)
            Input vector.

    Returns:
        np.ndarray
            Normalized vector (length 1).
    """
    return v / np.linalg.norm(v)


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute a rotation matrix that aligns vec1 to vec2 using Rodrigues' formula.

    Handles:
        - Identical vectors (returns identity)
        - Opposite vectors (returns 180° rotation around perpendicular axis)

    Parameters:
        vec1 : array-like, shape (3,)
            Initial vector.
        vec2 : array-like, shape (3,)
            Target vector.

    Returns:
        np.ndarray, shape (3, 3)
            Rotation matrix mapping vec1 → vec2.
    """
    a = normalize(vec1)
    b = normalize(vec2)
    cross = np.cross(a, b)
    dot = np.dot(a, b)
    if np.isclose(dot, 1.0):
        return np.eye(3)  # Already aligned
    if np.isclose(dot, -1.0):
        # 180-degree rotation around any perpendicular axis
        perp = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        axis = normalize(np.cross(a, perp))
        angle = np.pi
    else:
        axis = normalize(cross)
        angle = np.arccos(dot)

    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def rotate_volume_diff(volume, radii, r_vector, L_vectors):
    """
    Apply a twist deformation to a 3D volume based on varying spin direction vectors.

    The twist depends on radius: if L(r) aligns with the z-axis, no twist is applied.
    Misalignment induces a local rotation to align L(r) with ẑ.

    Parameters:
        volume    : np.ndarray, shape (Z, Y, X)
            Input 3D scalar field.
        radii     : np.ndarray, shape (Z, Y, X)
            Radius per voxel from the volume center.
        r_vector  : np.ndarray, shape (R,)
            Radius values where spin direction vectors are defined.
        L_vectors : np.ndarray, shape (3, R)
            Spin direction vectors corresponding to each r in `r_vector`.

    Returns:
        np.ndarray
            Deformed 3D volume (same shape as input).
    """
    shape_z, shape_y, shape_x = volume.shape
    center_x = (shape_x - 1) / 2.0
    center_y = (shape_y - 1) / 2.0
    center_z = (shape_z - 1) / 2.0

    # Create grid of centered voxel coordinates
    zz, yy, xx = np.indices((shape_z, shape_y, shape_x))
    x = xx.ravel().astype(np.float32) - center_x
    y = yy.ravel().astype(np.float32) - center_y
    z = zz.ravel().astype(np.float32) - center_z
    coords = np.stack([x, y, z], axis=1)  # Shape (N, 3)

    r_flat = radii.ravel()

    # Interpolate spin direction vectors at each voxel's radius
    interp = interp1d(r_vector, L_vectors, kind='linear', axis=1,
                      bounds_error=False, fill_value='extrapolate')
    L_interp = interp(r_flat).T  # Shape (N, 3)

    # Normalize spin direction vectors
    norms = np.linalg.norm(L_interp, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    L_unit = L_interp / norms

    # Reference axis (ẑ)
    z_axis = np.array([0.0, 0.0, 1.0])
    z_broadcast = np.tile(z_axis, (L_unit.shape[0], 1))

    # Compute rotation angles and axes
    dot = np.clip(np.sum(L_unit * z_broadcast, axis=1), -1.0, 1.0)
    angle = np.arccos(dot)
    axis = np.cross(z_broadcast, L_unit)

    # Normalize rotation axes
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    valid = axis_norm[:, 0] > 1e-8
    axis[valid] /= axis_norm[valid]

    # Rodrigues rotation for all points
    v = coords
    k = axis
    k_cross_v = np.cross(k, v)
    k_dot_v = np.sum(k * v, axis=1)

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    v_rot = (
        v * cos_theta[:, np.newaxis]
        + k_cross_v * sin_theta[:, np.newaxis]
        + k * (k_dot_v * (1.0 - cos_theta))[:, np.newaxis]
    )

    # Shift back to original coordinate frame
    v_rot[:, 0] += center_x
    v_rot[:, 1] += center_y
    v_rot[:, 2] += center_z

    # Prepare for interpolation
    coords_x = v_rot[:, 0].reshape((shape_z, shape_y, shape_x))
    coords_y = v_rot[:, 1].reshape((shape_z, shape_y, shape_x))
    coords_z = v_rot[:, 2].reshape((shape_z, shape_y, shape_x))
    coords_interp = np.stack([coords_z, coords_y, coords_x], axis=0)

    # Interpolate original volume at rotated coordinates
    deformed = map_coordinates(volume, coords_interp, order=1, mode='constant', cval=0.0)

    return deformed


def rotate_volume_solid(volume, vector):
    """
    Apply a solid-body rotation to a 3D volume to align the z-axis with `vector`.

    Parameters:
        volume : np.ndarray, shape (Z, Y, X)
            Input 3D scalar field.
        vector : array-like, shape (3,)
            Target vector for z-axis alignment.

    Returns:
        np.ndarray
            Rotated 3D volume.
    """
    shape = volume.shape
    center = np.array(shape) / 2

    # Compute rotation matrix and its inverse
    R = rotation_matrix_from_vectors([0, 0, 1], vector)
    inv_R = np.linalg.inv(R)

    # affine_transform applies the inverse rotation
    rotated = affine_transform(volume, inv_R, offset=np.dot(inv_R, -center) + center, order=1)

    return rotated


def compute_spin_vector(points, dt=1.0):
    """
    Compute the spin vector over time:
        L(t) = r(t) × v(t)

    Parameters:
        points : np.ndarray, shape (3, N)
            Position vectors over time.
        dt     : float, optional
            Time step between samples (default: 1.0).

    Returns:
        np.ndarray, shape (3, N)
            Spin vectors at each time step.
    """
    r = points
    v = np.gradient(r, dt, axis=1)  # Velocity via numerical derivative
    L = np.cross(r.T, v.T).T        # Cross product per timestep

    return L


def make_grid(R, n):
    """
    Create a 3D spherical coordinate grid.

    Parameters:
        R : float
            Maximum radius of the grid in Cartesian units.
        n : int
            Number of grid points along each axis.

    Returns:
        tuple of np.ndarray:
            x, y, z   : Cartesian coordinates (n×n×n)
            r         : Spherical radius
            th        : Polar angle (θ)
            phi       : Azimuthal angle (φ)
    """
    # Cartesian grid
    lin = np.linspace(-R, R, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing='ij')

    # Spherical grid
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(np.clip(z / r, -1.0, 1.0))  # Clip for numerical safety
    phi = np.arctan2(y, x)
    th[np.isnan(th)] = 0  # Handle origin singularity

    return x, y, z, r, th, phi

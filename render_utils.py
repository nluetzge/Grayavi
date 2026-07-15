# IDENT			render_utils.py
# LANGUAGE		Python
# AUTHOR		N. LUETZGENDORF
# PURPOSE	    Utility functions for grayavi.py for anything related to rendering with MayaVi.
#
# VERSION
# 1.0.0 12.08.2025 NL Creation
# ===============================================================
# Imports
# ===============================================================
import numpy as np
import matplotlib.pyplot as plt

from mayavi import mlab
from tvtk.api import tvtk
from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction

import cv2

from math_utils import map_range
# ===============================================================
# Functions
# ===============================================================


def render_waveform_matplotlib(time_vector, waveform_td, index,
                               input_dir="frames", output_dir="frames_combined"):
    """
    Combine a Mayavi-rendered frame with an overlaid time-domain waveform using Matplotlib.

    This function loads a pre-rendered frame (PNG) from a Mayavi visualization,
    displays it as the background in a Matplotlib figure, and overlays the
    corresponding time-domain waveform at the bottom of the image. The waveform
    is drawn with partial transparency for the full signal and full opacity for
    the portion up to the given frame index.

    Parameters
    ----------
    time_vector : array_like
        1D array of time values corresponding to the waveform samples.
    waveform_td : array_like
        1D array of complex or real time-domain waveform values. Only the real
        part is plotted.
    index : int
        Frame index. Used to determine which portion of the waveform is fully
        opaque and to select the corresponding Mayavi frame image.
    input_dir : str, optional
        Directory containing the Mayavi-rendered frames. Default is ``"frames"``.
    output_dir : str, optional
        Directory where the combined output frames will be saved.
        Default is ``"frames_combined"``.

    Notes
    -----
    - The background Mayavi image is assumed to be named using the pattern
      ``frame_####.png``, where `####` is zero-padded to 4 digits.
    - The waveform overlay is drawn in white with alpha blending:
      alpha=0.5 for the full waveform, alpha=1 for the portion up to `index`.
    - The output image is saved with the same filename pattern in `output_dir`.

    See Also
    --------
    matplotlib.pyplot.subplots : Create a new figure with subplots.
    cv2.imread : Read an image from a file.
    cv2.cvtColor : Convert image color spaces.

    Examples
    --------
    >>> render_waveform_matplotlib(t, h_td, 25, input_dir="frames", output_dir="frames_combined")
    # Produces "frames_combined/frame_0025.png" with Mayavi background and waveform overlay.
    """
    # Load the rendered Mayavi image
    mayavi_img = cv2.imread(f"{input_dir}/frame_{index:04d}.png")
    mayavi_img = cv2.cvtColor(mayavi_img, cv2.COLOR_BGR2RGB)  # Convert for matplotlib
    height, width = mayavi_img.shape[:2]

    # Set up a Matplotlib figure the same size
    dpi = 70
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')

    # Show Mayavi frame
    ax.imshow(mayavi_img)
    ax.axis("off")

    # Overlay the waveform at the bottom
    ax2 = fig.add_axes((0.1, 0.05, 0.8, 0.15))  # [left, bottom, width, height]
    ax2.plot(time_vector, waveform_td.real, color='white', lw=2, alpha=0.5)
    ax2.plot(time_vector[:index], waveform_td.real[:index], color='white', lw=2, alpha=1)
    ax2.set_facecolor('black')
    ax2.set_xlim(time_vector[0], time_vector[-1])
    ax2.axis("off")
    fig.tight_layout()

    # Save combined image
    fig.savefig(f"{output_dir}/frame_{index:04d}.png")
    plt.close(fig)


def plot_big_bh(center=(0., 0., 0.), size=1.0, color=(1., 1., 1.)):
    """
    Plot a large, shiny sphere representing a black hole in a Mayavi scene.

    This function creates a high-resolution sphere mesh using a parametric
    grid in spherical coordinates, then renders it with Mayavi's `mlab.mesh`.
    The sphere is given a reflective, glossy appearance.

    Parameters
    ----------
    center : tuple of float, optional
        (x, y, z) coordinates of the sphere's center.
        Defaults to (0., 0., 0.).
    size : float, optional
        Radius of the sphere. Defaults to 1.0.
    color : tuple of float, optional
        RGB color of the sphere, with each component in the range [0, 1].
        Defaults to (1., 1., 1.) for white.

    Returns
    -------
    sphere : mayavi.modules.surface.Surface
        The Mayavi mesh object representing the sphere.

    Notes
    -----
    - The sphere mesh resolution is set to 100 points in both polar and
      azimuthal directions for smoothness.
    - The `specular` and `specular_power` properties are adjusted to
      create a shiny, reflective surface.

    Examples
    --------
    >>> from mayavi import mlab
    >>> mlab.figure(bgcolor=(0, 0, 0))
    >>> plot_big_bh(center=(0, 0, 0), size=2.0, color=(0.5, 0.5, 1.0))
    >>> mlab.show()
    """
    # Generate high-resolution sphere mesh
    phi, theta = np.mgrid[0:np.pi:100j, 0:2 * np.pi:100j]  # More points for smoothness
    xb = size * np.sin(phi) * np.cos(theta) + center[0]
    yb = size * np.sin(phi) * np.sin(theta) + center[1]
    zb = size * np.cos(phi) + center[2]

    # Plot the sphere mesh
    sphere = mlab.mesh(xb, yb, zb, color=color)

    # Make it shiny
    sphere.actor.property.specular = 1.0  # Maximum shininess
    sphere.actor.property.specular_power = 100  # Sharper reflections
    return sphere


def plot_orbit(i, points, center=(0., 0., 0.),
               bh_color=(1., 1., 1.), tube_radius=1e-1,
               orbit_color=(1., 1., 1.), trail_length=50, size=0.1):
    """
    Plot an orbiting body with a fading trail in a Mayavi 3D scene.

    This function renders a point representing the orbiting body at a given
    time step, along with a trail showing its recent path. The trail fades
    from opaque to transparent over the specified number of previous points.

    Parameters
    ----------
    i : int
        Current time-step index to plot.
    points : tuple of array_like
        A tuple (x, y, z) containing the coordinates of the orbit path over time.
        Each array must be 1D and of the same length.
    center : tuple of float, optional
        (x, y, z) coordinates for shifting the entire orbit. Defaults to (0., 0., 0.).
    bh_color : tuple of float, optional
        RGB color of the orbiting body, each component in [0, 1]. Defaults to white.
    tube_radius : float, optional
        Radius of the orbit trail tube. Defaults to 0.1.
    orbit_color : tuple of float, optional
        RGB color of the orbit trail, each component in [0, 1]. Defaults to white.
    trail_length : int, optional
        Number of previous points to include in the fading trail. Defaults to 50.
    size : float, optional
        Scale factor for the orbiting body marker. Defaults to 0.1.

    Returns
    -------
    None
        The plotted objects are added to the active Mayavi scene but not returned.

    Notes
    -----
    - The fading effect is achieved by decreasing the trail segment opacity
      linearly from 1.0 (opaque) to 0.0 (transparent) along the last `trail_length` points.
    - The function must be called iteratively for animation over multiple time steps.

    Examples
    --------
    >>> from mayavi import mlab
    >>> import numpy as np
    >>> t = np.linspace(0, 2*np.pi, 200)
    >>> x = np.cos(t)
    >>> y = np.sin(t)
    >>> z = np.zeros_like(t)
    >>> mlab.figure(bgcolor=(0, 0, 0))
    >>> for i in range(len(t)):
    ...     plot_orbit(i, (x, y, z), bh_color=(1, 0, 0), orbit_color=(0, 1, 0))
    >>> mlab.show()
    """
    # Shift orbit points by the given center offset
    x1, y1, z1 = points
    x1 = x1 + center[0]
    y1 = y1 + center[1]
    z1 = z1 + center[2]

    # Plot the orbiting body at the current position
    mlab.points3d(x1[i], y1[i], z1[i], scale_factor=size, color=bh_color)

    # Plot the fading trail
    if i > 0:
        start = max(0, i - trail_length)
        end = i
        length = end - start
        alphas = np.linspace(0, 1, length)
        for j, alpha in zip(np.arange(start, end, 1), alphas):
            mlab.plot3d(
                x1[j:j+2], y1[j:j+2], z1[j:j+2],
                opacity=alpha,
                tube_radius=tube_radius,
                color=orbit_color
            )


def plot_sky(pos=(0., 0., 0.), size=2e11, im='8k_stars_milky_way.jpg', resolution=180):
    """
    Create a textured celestial sphere for a Mayavi/TVTK scene.

    This function generates a large sphere textured with a starfield image
    to act as a background sky. The texture is applied to the **inside** of
    the sphere by flipping its normals, creating the illusion of being
    surrounded by space.

    Parameters
    ----------
    pos : tuple of float, optional
        (x, y, z) coordinates for the center of the sphere.
        Defaults to (0., 0., 0.).
    size : float, optional
        Radius of the sphere in scene units. Defaults to 2e11.
    im : str, optional
        Path to the JPEG image file to use as the texture.
        Defaults to '8k_stars_milky_way.jpg'.
    resolution : int, optional
        Number of subdivisions in both polar and azimuthal directions
        (`theta_resolution` and `phi_resolution`). Higher values
        increase visual quality at the cost of performance.
        Defaults to 180.

    Returns
    -------
    sphere_actor : tvtk.Actor
        TVTK actor representing the textured sphere.
        Add to a Mayavi scene with:
        ``mlab.gcf().scene.add_actor(sphere_actor)``.

    Notes
    -----
    - `frontface_culling` is enabled and `backface_culling` is disabled so
      that the texture is visible from the inside of the sphere.
    - The `specular` and `specular_power` properties are set for subtle
      reflective highlights.
    - The default `resolution` of 180 creates a smooth sphere suitable
      for high-quality backgrounds.

    Examples
    --------
    >>> from mayavi import mlab
    >>> sky = plot_sky(size=1e3, im='milky_way.jpg', resolution=100)
    >>> mlab.gcf().scene.add_actor(sky)
    >>> mlab.show()
    """
    x, y, z = pos
    img = tvtk.JPEGReader()
    img.file_name = im
    # Map the texture
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=0)

    # Create the sphere
    sphere = tvtk.TexturedSphereSource(radius=size, theta_resolution=resolution,
                                       phi_resolution=resolution)
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)

    # Position the sphere
    sphere_actor.position = x, y, z
    # Flip normals to display texture on the inside
    sphere_actor.property.frontface_culling = True
    sphere_actor.property.backface_culling = False

    # Set shine and material properties
    sphere_actor.property.specular = 1
    sphere_actor.property.specular_power = 30
    return sphere_actor


def transfer_functions(pos_first_peak, pos_last_peak, cmap_name='turbo', n_peaks=10,
                       logscale=False, opmin=0.02, opmax=0.6, width=0.01):
    """
    Create opacity and color transfer functions for volumetric rendering.

    Generates a set of narrow opacity spikes and corresponding RGB colors
    at evenly or logarithmically spaced positions between the first and last
    peak values. This is useful for rendering layered shell-like structures
    in volume visualizations (e.g., astrophysical waveforms).

    Parameters
    ----------
    pos_first_peak : float
        Position of the first peak (minimum contour value).
    pos_last_peak : float
        Position of the last peak (maximum contour value).
    cmap_name : str, optional
        Name of the Matplotlib colormap to use for the color transfer function.
        Default is 'turbo'.
    n_peaks : int, optional
        Number of peaks (contours) to generate. Default is 10.
    logscale : bool, optional
        If True, contour positions are spaced geometrically (logarithmic scale);
        otherwise they are spaced linearly. Default is False.
    opmin : float, optional
        Minimum opacity value at the first peak. Default is 0.02.
    opmax : float, optional
        Maximum opacity value at the last peak. Default is 0.6.
    width : float, optional
        Relative width of each opacity spike, expressed as a fraction of the
        total range between `pos_first_peak` and `pos_last_peak`. Default is 0.1.
        Note: Do not change this unless you really think it is needed it just messes with the
        light of the scene.

    Returns
    -------
    otf : PiecewiseFunction
        Opacity transfer function with narrow spikes at each contour position.
    ctf : ColorTransferFunction
        Color transfer function mapping contour positions to RGB values.

    Notes
    -----
    The opacity spikes are triangular, centered at each contour position,
    with zero opacity outside the spike width.

    Examples
    --------
    >>> otf, ctf = transfer_functions(0.1, 1.0, cmap_name='viridis', n_peaks=5, width=0.05)
    >>> # Pass otf and ctf to a volume rendering pipeline
    """

    if logscale:
        contours = np.geomspace(pos_first_peak, pos_last_peak, n_peaks)
    else:
        contours = np.linspace(pos_first_peak, pos_last_peak, n_peaks)

    cmap = plt.get_cmap(cmap_name)

    # Opacity transfer function
    otf = PiecewiseFunction()

    # Color transfer function
    ctf = ColorTransferFunction()

    # Width of spikes as a fraction of the total width
    spike_width = width * (pos_last_peak - pos_first_peak)  # shell thickness

    for c in contours:
        # Color transfer
        ci = map_range(c, pos_first_peak, pos_last_peak)  # normalized colormap position
        color = cmap(ci)[:3]
        ctf.add_rgb_point(c, *color)

        # Opacity transfer — sharp spike centered at c
        opacity = map_range(c, pos_first_peak, pos_last_peak, b1=opmin, b2=opmax)
        otf.add_point(c - spike_width, 0.0)
        otf.add_point(c, opacity)
        otf.add_point(c + spike_width, 0.0)

    return otf, ctf


def replicate_on_screen_lights(offscreen_renderer):
    """
    Recreate on-screen lighting configuration in an off-screen renderer.

    When performing off-screen rendering in VTK/Mayavi, the default
    on-screen lighting setup is not preserved. This function removes
    any existing lights from the given renderer and adds a set of
    predefined lights to mimic a typical on-screen lighting arrangement.

    Parameters
    ----------
    offscreen_renderer : tvtk.Renderer
        The off-screen renderer instance in which to replicate the lights.

    Notes
    -----
    - All existing lights in the renderer are removed before adding new ones.
    - Four lights are created:
        1. A headlight with zero intensity (placeholder for completeness).
        2–4. Camera lights positioned around the scene with varying intensities.
    - Light properties such as position, focal point, colors, and intensity
      are hardcoded based on a typical on-screen configuration.

    Examples
    --------
    >>> from tvtk.api import tvtk
    >>> renderer = tvtk.Renderer()
    >>> replicate_on_screen_lights(renderer)
    >>> len(renderer.lights)  # Check that lights were added
    4
    """

    # Unfortunately when we do off-screen rendering all our pretty lights are gone and we
    # need to build them again ourselves

    # Remove all lights
    while offscreen_renderer.lights.number_of_items > 0:
        light = offscreen_renderer.lights.get_item_as_object(0)
        offscreen_renderer.remove_light(light)

    # Define your lights based on your on-screen data:
    lights_data = [
        {
            'light_type': 'headlight',
            'position': (1.2246468e-14, -200.0, 1.2246468e-14),
            'focal_point': (0, 0, 0),
            'diffuse_color': (1.0, 1.0, 1.0),
            'specular_color': (1.0, 1.0, 1.0),
            'ambient_color': (0.0, 0.0, 0.0),
            'intensity': 0.0,
            'switch': True,
        },
        {
            'light_type': 'camera_light',
            'position': (0.5, 0.70710678, 0.5),
            'focal_point': (0, 0, 0),
            'diffuse_color': (1.0, 1.0, 1.0),
            'specular_color': (1.0, 1.0, 1.0),
            'ambient_color': (0.0, 0.0, 0.0),
            'intensity': 1.0,
            'switch': True,
        },
        {
            'light_type': 'camera_light',
            'position': (-0.75, -0.5, 0.4330127),
            'focal_point': (0, 0, 0),
            'diffuse_color': (1.0, 1.0, 1.0),
            'specular_color': (1.0, 1.0, 1.0),
            'ambient_color': (0.0, 0.0, 0.0),
            'intensity': 1,
            'switch': True,
        },
        {
            'light_type': 'camera_light',
            'position': (0.75, -0.5, 0.4330127),
            'focal_point': (0, 0, 0),
            'diffuse_color': (1.0, 1.0, 1.0),
            'specular_color': (1.0, 1.0, 1.0),
            'ambient_color': (0.0, 0.0, 0.0),
            'intensity': 1,
            'switch': True,
        },
    ]

    for ld in lights_data:
        if not ld['switch']:
            continue  # skip lights that are off

        light = tvtk.Light()
        light.light_type = ld['light_type']
        light.position = ld['position']
        light.focal_point = ld['focal_point']
        light.diffuse_color = ld['diffuse_color']
        light.specular_color = ld['specular_color']
        light.ambient_color = ld['ambient_color']
        light.intensity = ld['intensity']
        light.switch = True
        offscreen_renderer.add_light(light)


def render_volume(x, y, z, field, otf=None, ctf=None, points=None, points2=None,
                  index=None, zoom=300, angle=-90, elevation=90, xdim=1350,
                  ydim=800, add_background=False, background_file=None, background_color=(0, 0, 0),
                  objects=True, object_color=(1, 1, 1), spin_vector=False, L=None, offscreen=True):
    """
    Render a volumetric scalar field with optional orbits, spin vectors,
    background imagery, and off-screen lighting corrections using Mayavi.

    This function builds a full 3D rendering pipeline from a scalar field,
    applying custom color and opacity transfer functions, camera parameters,
    and additional scene elements like black holes, orbital trajectories,
    and background spheres. It supports both on-screen and off-screen rendering
    for scientific visualization and movie frame generation.

    Parameters
    ----------
    x, y, z : ndarray
        3D coordinate grids defining the spatial positions of the scalar field.
    field : ndarray
        Scalar field values to be rendered (same shape as `x`, `y`, `z`).
    otf : PiecewiseFunction, optional
        Custom opacity transfer function. Overrides the default if provided.
    ctf : ColorTransferFunction, optional
        Custom color transfer function. Overrides the default if provided.
    points : ndarray, optional
        Array of (N, 3) positions for the primary object’s trajectory or black hole.
    points2 : ndarray, optional
        Array of (N, 3) positions for a secondary trajectory (e.g., second black hole).
    index : int, optional
        Time-step index into the trajectory arrays used to plot the current positions.
    zoom : float, optional
        Camera distance from the scene center. Default is 300.
    angle : float, optional
        Azimuthal view angle in degrees. Default is -90.
    elevation : float, optional
        Elevation view angle in degrees. Default is 90.
    xdim : int, optional
        Width of the rendering window in pixels. Default is 1350.
    ydim : int, optional
        Height of the rendering window in pixels. Default is 800.
    add_background : bool, optional
        If True, adds a background sphere using a texture image.
    background_file : str, optional
        Path to the background image file (used only if `add_background=True`).
    background_color : tuple of float, optional
        RGB tuple for background color. Default is (0, 0, 0).
    objects : bool, optional
        If True, plots black holes and/or orbital trajectories. Default is True.
    object_color : tuple of float, optional
        RGB color for rendered objects (black holes, orbits, and spin vectors).
        Default is white (1, 1, 1).
    spin_vector : bool, optional
        If True, plots a 3D vector arrow representing the spin direction.
    L : array-like, optional
        3-component spin vector used when `spin_vector=True`.
        The vector’s magnitude and direction determine the arrow’s length and orientation.
    offscreen : bool, optional
        If True, enables off-screen rendering and restores lighting to match
        the on-screen appearance (avoiding dark renders).

    Returns
    -------
    fig : mayavi.core.scene.Scene
        Mayavi scene object containing the fully rendered visualization.

    Notes
    -----
    - The function automatically configures lighting and shading to produce
      consistent brightness in both on-screen and off-screen rendering modes.
    - Camera positioning is handled via `mlab.view` using azimuth, elevation,
      and zoom for reproducible viewpoints.
    - If both `points` and `points2` are provided, two orbital trajectories
      are plotted; otherwise, a single black hole and one orbit are shown.
    - `spin_vector` is primarily used for debugging or visualizing spin direction
      at a given simulation time step.

    Examples
    --------
    >>> fig = render_volume(
    ...     x, y, z, field,
    ...     otf=my_otf, ctf=my_ctf,
    ...     points=bh1_orbit, points2=bh2_orbit,
    ...     index=42, add_background=True,
    ...     background_file='8k_stars.jpg',
    ...     spin_vector=True, L=[0, 0, 1]
    ... )
    >>> mlab.savefig('frame_0042.png', figure=fig)
    """

    fig = mlab.figure(size=(xdim, ydim), bgcolor=background_color)

    if offscreen:
        fig.scene.off_screen_rendering = True

    if add_background:
        sky = plot_sky(size=2.5e2, pos=[0, 0, 0], im=background_file)
        fig.scene.add_actor(sky)

    # Create the scalar field source
    src = mlab.pipeline.scalar_field(x, y, z, field)
    src.update_image_data = True

    # This was just for debugging to plot the spin vector, then you also need to provide L,
    # spin at that time step
    if spin_vector:
        mlab.quiver3d(0, 0, 0, L[0], L[1], L[2], scale_factor=1.0, color=object_color)

    # Volume rendering pipeline
    volume = mlab.pipeline.volume(src)

    # Opacity transfer function
    if otf is not None:
        volume._volume_property.set_scalar_opacity(otf)

    # Color transfer function (blue -> black -> red)
    if ctf is not None:
        volume._volume_property.set_color(ctf)

    # Lighting
    volume._volume_property.shade = True
    volume._volume_property.ambient = 0.1
    volume._volume_property.diffuse = 0.9
    volume._volume_property.specular = 0.3
    volume._volume_property.specular_power = 10

    # Plot Bhs and trajectory
    if objects:
        if points is not None:
            if points2 is not None:
                plot_orbit(index, points, size=1, bh_color=object_color, orbit_color=object_color)
                plot_orbit(index, points2, size=1, bh_color=object_color, orbit_color=object_color)
            else:
                plot_big_bh(size=0.8, color=object_color)
                plot_orbit(index, points, size=0.5, bh_color=object_color, orbit_color=object_color)

    # Set the view (azimuth=0, elevation=90 is edge on)
    mlab.view(azimuth=angle, elevation=elevation, distance=zoom)

    # Add lights back for off-screen rendering
    if offscreen:
        replicate_on_screen_lights(fig.scene.renderer)

    return fig

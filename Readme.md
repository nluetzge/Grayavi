![logo](./logo/grayavi_logo.png?raw=true "Optional Title")

Gayavi is a Python package for creating three-dimensional visualisations of gravitational-wave signals using Mayavi. The package renders spin-weighted spherical harmonic representations of waveforms as volumetric structures and can optionally combine the 3D rendering with the corresponding time-domain waveform to produce animations suitable for scientific visualisation, outreach, and presentations. It supports a range of LISA source classes including Galactic Binaries (GBs), Massive Black Hole Binaries (MBHBs), and Extreme Mass Ratio Inspirals (EMRIs).

Features
--------

* 3D volumetric rendering of gravitational-wave modes using spin-weighted spherical harmonics.
* Support for Galactic Binaries (GB), Massive Black Hole Binaries (MBHB), and Extreme Mass Ratio Inspirals (EMRI).
* Optional rendering of binary trajectories and spin vectors.
* Optional overlay of the time-domain waveform beneath the 3D visualisation.
* Configurable camera zoom, rotation, colour maps, opacity transfer functions, and background imagery.
* Automatic filename generation based on selected rendering parameters.
* Export of individual frames and MP4 videos using FFmpeg.

Package Structure
-----------------


    Gayavi/
    ├── auxfiles/
    │   └── 8k_stars_milky_way.jpg
    │
    ├── configs/
    │   ├── EMRI_config.yaml
    │   ├── GB_config.yaml
    │   └── MBHB_config.yaml
    │
    ├── waveforms/
    │   ├── emri_waveform_and_trajectory_im.h5
    │   ├── gb_waveform.h5
    │   └── mbhb_waveform_q1.h5
    │
    ├── output/
    │   ├── frames/
    │   ├── frames_combined/
    │   └── videos/
    │
    ├── data_utils.py
    ├── math_utils.py
    ├── render_utils.py
    └── grayavi.py

Main Components
---------------

``grayavi.py``


Main driver script responsible for:

* Loading a YAML configuration file.
* Reading waveform data from an HDF5 file.
* Generating spin-weighted spherical harmonic representations.
* Rendering 3D volumetric visualisations.
* Producing frame images.
* Combining waveform plots with rendered frames.
* Generating MP4 videos using FFmpeg.

``data_utils.py``


Utility functions for:

* Reading waveform data from HDF5 files.
* Generating spin-weighted spherical harmonic grids.
* Loading and managing configuration files.
* Automatic output filename generation.
* Creation of test waveforms.

``math_utils.py``


Mathematical helper functions including:

* Coordinate transformations.
* Smooth activation and deactivation functions.
* Volume rotations.
* Spin-vector calculations.
* Grid generation.

``render_utils.py``


Rendering-related functionality including:

* Volumetric rendering with Mayavi.
* Orbit and black-hole visualisation.
* Background sky rendering.
* Transfer function generation.
* Combination of waveform and rendering outputs.

Configuration Files
-------------------

All rendering parameters are controlled through YAML configuration files located in the ``configs`` directory.

Available examples include:

* ``GB_config.yaml``
* ``MBHB_config.yaml``
* ``EMRI_config.yaml``

Configuration files control:

* Input waveform file.
* Number of frames.
* Camera settings.
* Rendering dimensions.
* Frame rate.
* Colour maps and opacity settings.
* Background rendering.
* Waveform overlays.
* Output directories.
* Rendering frame ranges.

Rendering Large Animations
--------------------------

Rendering very large numbers of frames in a single run can cause Mayavi to become slow or unresponsive. For this reason it is strongly recommended to render animations in batches of approximately 300–500 frames.

The parameters


    start_frame
    end_frame

allow only a subset of the full animation to be rendered during a given run. These parameters exist specifically to support batch rendering of long animations.

For example, a 1000-frame animation can be rendered in two batches:

Batch 1:


    start_frame: 0
    end_frame: 499

Batch 2:


    start_frame: 500
    end_frame: 999

After all batches have been rendered, the complete frame set can be combined into a single video. This workflow is recommended for all long renderings.

Typical Workflow
----------------

1. Select the appropriate configuration file and adjust the required parameters.


    waveform_filename:
        value: "waveforms/gb_waveform.h5"

    n_frames:
        value: 1000

    start_frame:
        value: 0

    end_frame:
        value: 499

2. Run the renderer.


    python grayavi.py

3. Continue rendering additional frame batches by updating ``start_frame`` and ``end_frame``.

4. Once all frames have been generated, set


    video_only:
        value: true

and run the script again to create the final MP4 file.

Input Data
----------

Gayavi expects waveform data stored in HDF5 format (``.h5``). Depending on the source type, files may additionally contain trajectory information that can be used to render orbital motion and spin evolution.

Output Products
---------------

Generated products are written to


    output/
    ├── frames/
    ├── frames_combined/
    └── videos/

where

* ``frames`` contains the raw Mayavi renderings.
* ``frames_combined`` contains renderings combined with waveform plots.
* ``videos`` contains the final MP4 animations.

Dependencies
------------

Gayavi relies on the following Python packages:

* NumPy
* SciPy
* Matplotlib
* Mayavi
* TVTK
* OpenCV
* h5py
* quaternionic
* spherical
* PyYAML

In addition, FFmpeg is required for video generation.

Installation
------------

It is recommended to install the required Python packages in a dedicated virtual environment.


    pip install numpy scipy matplotlib mayavi opencv-python h5py quaternionic spherical pyyaml

FFmpeg must also be installed and accessible from the command line.

Author
------

Nora Luetzgendorf

Gayavi was developed for the visualisation of gravitational-wave sources and for communicating LISA science through high-quality scientific animations.

![logo](graphics/grayavi_logo.png?raw=true "Optional Title")

Gayavi is a Python package for creating three-dimensional visualisations of gravitational-wave signals using Mayavi. The package renders spin-weighted spherical harmonic representations of waveforms as volumetric structures and can optionally combine the 3D rendering with the corresponding time-domain waveform to produce animations suitable for scientific visualisation, outreach, and presentations. It supports a range of LISA source classes including Galactic Binaries (GBs), Massive Black Hole Binaries (MBHBs), and Extreme Mass Ratio Inspirals (EMRIs).

Installation
------------

It is recommended to install the required Python packages in a dedicated virtual environment. 
This installation is from October 2025. I hope it still works.


    conda create -n mayavi python=3.11
    conda install pyqt=5 qt-main=5.15 mayavi
    pip install opencv-python
    conda install pyyaml scipy quaternionic spherical h5py 

FFmpeg must also be installed and accessible from the command line.

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
* FFmpeg

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

Select the appropriate configuration file and adjust the required parameters.

    
    waveform_filename:
        value: "waveforms/gb_waveform.h5"

    n_frames:
        value: 1000

    start_frame:
        value: 0

    end_frame:
        value: 499

Run the renderer with your desired config file.


    python grayavi.py configs/GB_config.yaml 

Continue rendering additional frame batches by updating ``start_frame`` and ``end_frame``.

Input Data
----------

Gayavi expects all datasets to be located at the root level of the HDF5 file.

A minimal file has the structure:

.. code-block:: text

    /
    ├── time_vector      (N,)
    ├── waveform_td      (2, N)
    ├── traj_x           (N,)
    ├── traj_y           (N,)
    └── traj_z           (N,)

For binary systems a second trajectory can be supplied:

.. code-block:: text

    /
    ├── time_vector      (N,)
    ├── waveform_td      (2, N)
    ├── traj_x           (N,)
    ├── traj_y           (N,)
    ├── traj_z           (N,)
    ├── traj_x2          (N,)
    ├── traj_y2          (N,)
    └── traj_z2          (N,)

All arrays must contain at least ``n_frames`` samples, as specified in the configuration file.

The ``waveform_td`` dataset contains the real and imaginary parts of the waveform:

* ``waveform_td[0, :]`` = real part
* ``waveform_td[1, :]`` = imaginary part


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


Author
------

Nora Lützgendorf

Grayavi was developed based on [gwpv](https://github.com/nilsvu/gwpv) which uses the same techinque but with a different graphical engine ParaView. I found it hard to install ParaView on a Mac so I decided to create a new code using something more native (Mayavi). The credit for the algorithm should go to [Nils Vu](https://github.com/nilsvu) since I would have not managed without looking at his great code. 


![banner](graphics/colormaps.png?raw=true "Optional Title")

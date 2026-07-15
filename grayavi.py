# IDENT			code.py
# LANGUAGE		Python
# AUTHOR		N. LUETZGENDORF
# PURPOSE		
#
# VERSION
# 1.0.0 DD.MM.YYYY NL Creation
# ===============================================================
# Imports
# ===============================================================
import os
import subprocess
import argparse
import numpy as np

from mayavi import mlab

from math_utils import map_range, rotate_volume_solid, rotate_volume_diff, make_grid
from render_utils import render_waveform_matplotlib, transfer_functions, render_volume
from data_utils import read_waveform, swsh_grid, Config

# ===============================================================
# Main
# ===============================================================
# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Render gravitational wave visualization frames/video.")
parser.add_argument(
    "config",
    nargs="?",
    default="configs/GB_config.yaml",
    help="Path to the config YAML file (default: %(default)s)"
)
args = parser.parse_args()

# Load configuration
config = Config(args.config)

# Check if output directory exists
os.makedirs(config["output_dir"], exist_ok=True)
frames_dir = os.path.join(config["output_dir"], "frames")
frames_comb_dir = os.path.join(config["output_dir"], "frames_combined")
video_dir = os.path.join(config["output_dir"], "videos")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(frames_comb_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# If we only do the video we can skip all this
if not config["video_only"]:
    # Read waveform
    time_vector, waveform_td, L, points, points2 = read_waveform(config["waveform_filename"],
                                                                 config["n_frames"],
                                                                 config["ts"], config["rs"])

    # Grid setup
    x, y, z, r, th, phi = make_grid(config["R"], config["n"])

    # Create swsh grid (at the moment we only have 2,2)
    print("Creating the grid...")
    swsh_22 = swsh_grid(config["l"], config["m"], th, phi, r, config["ell_max"],
                        config["spin_weight"], config["size"], config["activation_offset"],
                        config["activation_width"], config["deactivation_width"])

    # Set up the transfer functions for plotting
    mode_max = np.max(np.abs(waveform_td))
    pos_first_peak, pos_last_peak = config["cmin"] * mode_max, config["cmax"] * mode_max
    otf, ctf = transfer_functions(pos_first_peak, pos_last_peak, cmap_name=config["cmap_name"],
                                  n_peaks=config["n_peaks"], opmin=config["opmin"],
                                  opmax=config["opmax"])

    # Create frames
    i = 0
    for i, t in zip(range(config["start_frame"], config["end_frame"]),
                    time_vector[config["start_frame"]:config["end_frame"]]):

        print("Rendering frame {:}/{:}".format(i, config["n_frames"]))

        # Interpolating the waveform along the 3D r scale
        phase = t - r + config["activation_offset"] * config["radial_scale"]
        mode_data = np.interp(phase, time_vector, waveform_td.real, right=0.0, left=0.0) \
            + 1j * np.interp(phase, time_vector, waveform_td.imag, right=0.0, left=0.0)

        # Multiplying with the swsh grid
        field_tmp = np.real(mode_data * swsh_22)

        # Rotating the spherical harmonics to align with spin
        # Solid rotation (would not recommend it looks weird)
        if config["spinrot"] == "solid":
            field = rotate_volume_solid(field_tmp, L[:, i])
        # Differential rotation of spin with time
        elif config["spinrot"] == "diff":
            field = rotate_volume_diff(field_tmp, phase, time_vector, L)
        # No rotation (when you know it is mostly planar), set spinrot to "none"
        else:
            field = field_tmp

        # Preparing zoom and rotation in case we have movement
        # Zoom
        if config["zoom"]:
            if config["zimin"] <= i <= config["zimax"]:
                zoom_in = map_range(i, config["zimin"], config["zimax"], b1=config["z1"],
                                    b2=config["z2"])
            elif i < config["zimin"]:
                zoom_in = config["z1"]
            elif i > config["zimax"]:
                zoom_in = config["z2"]
            else:
                print('Something went wrong with the zoom range, setting it to static zoom')
                zoom_in = config["zoom_static"]
        else:
            zoom_in = config["zoom_static"]
        # Rotate
        if config["rotate"]:
            if config["rimin"] <= i <= config["rimax"]:
                rot_ang = map_range(i, config["rimin"], config["rimax"], b1=config["r1"],
                                    b2=config["r2"])
            elif i < config["rimin"]:
                rot_ang = config["r1"]
            elif i > config["rimax"]:
                rot_ang = config["r2"]
            else:
                print('Something went wrong with the rotation range, setting it to static rotation')
                rot_ang = config["angle_static"]
        else:
            rot_ang = config["angle_static"]

        # Render the field (this is where magic happens
        fig = render_volume(x, y, z, field, otf=otf, ctf=ctf, points=points, points2=points2,
                            index=i, zoom=zoom_in, angle=rot_ang, elevation=config["elevation"],
                            xdim=config["xdim"], ydim=config["ydim"],
                            add_background=config["add_background"],
                            background_file=config["background_file"],
                            background_color=tuple(config["background_color"]),
                            objects=config["plot_bhs"], object_color=tuple(config["bh_color"]),
                            offscreen=True)


        # Save to file
        filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
        mlab.savefig(filename, figure=fig, magnification=config["mag"])

        mlab.close()

        # Render waveform if desired
        if config["add_waveform"]:
            render_waveform_matplotlib(time_vector, waveform_td, i,
                                       input_dir=frames_dir, output_dir=frames_comb_dir)
else:
    i = 0

# ===============================================================
# Create the video
# ===============================================================
# Create the name (from parameters)
filename = config.build_filename()

# Use combined frames if we are also plotting the waveform
if config["add_waveform"]:
    final_dir = frames_comb_dir
else:
    final_dir = frames_dir

# Only if we are at the end of we just do the video create do we do this step
if i == config["n_frames"] - 1 or config["video_only"]:
    print("Saving {:s}.mp4".format(filename))
    subprocess.run([
        "ffmpeg",
        "-framerate", str(config["framerate"]),
        "-pattern_type", "glob",
        "-i", f"{final_dir}/frame_*.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        f"{video_dir}/{filename}.mp4"
    ], check=True)

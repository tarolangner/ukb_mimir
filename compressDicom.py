import os
import sys
import io

import time

import torch

import zipfile
import pydicom

import numpy as np

import scipy.interpolate
import numba_interpolate

import skimage.measure 
from skimage import filters

import cv2
import nrrd

from torch.utils import data

c_out_pixel_spacing = np.array((2.23214293, 2.23214293, 3.))
c_resample_tolerance = 0.01 # Only interpolate voxels further off of the voxel grid than this

c_interpolate_seams = True # If yes, cut overlaps between stations to at most c_max_overlap and interpolate along them, otherwise cut at center of overlap
c_correct_intensity = True # If yes, apply intensity correction along overlap
c_max_overlap = 8 # Used in interpolation, any station overlaps are cut to be most this many voxels in size

c_trim_axial_slices = 4 # Trim this many axial slices from the output volume to remove folding artefacts

c_store_signals = True # If yes, store signal images

c_datatype_numpy = "float32" # See: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html


##### 
# Given the Dicom data of one subject as input zip: 
# Extract all water and fat signal slices
# Combine them to stations
# Resample and fuse them into complete volumes
# Format them into a two-dimensional RGB image
def fuseAndProjectDicom(input_path_zip):

    if not os.path.exists(input_path_zip):
        print("        Could not find input file {}".format(input_path_zip))
        return (None, False)
    
    # Extract all signal slices from Dicom
    (seg_voxel_data, seg_names, seg_positions, seg_pixel_spacings, seg_timestamps) = stationsFromDicom(input_path_zip)

    # Find water and fat signal station data
    (voxel_data_w, positions_w, pixel_spacings, timestamps_w) = extractStationsWithTag("_W", seg_names, seg_voxel_data, seg_positions, seg_pixel_spacings, seg_timestamps)
    (voxel_data_f, positions_f, _, timestamps_f)              = extractStationsWithTag("_F", seg_names, seg_voxel_data, seg_positions, seg_pixel_spacings, seg_timestamps)

    # Ensure that water and fat stations match in position and size and non-redundant
    (stations_consistent, voxel_data_w, voxel_data_f, positions, pixel_spacings) = ensureStationConsistency(voxel_data_w, voxel_data_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings)
    if not stations_consistent: 
        return (None, False)

    # Resample stations onto output volume voxel grid
    (voxel_data_w, _, _, _)          = resampleStations(voxel_data_w, positions, pixel_spacings)
    (voxel_data_f, W, W_end, W_size) = resampleStations(voxel_data_f, positions, pixel_spacings)

    # Cut station overlaps to at most c_max_overlap
    (_, _, _, _, voxel_data_w)                 = trimStationOverlaps(W, W_end, W_size, voxel_data_w)
    (overlaps, W, W_end, W_size, voxel_data_f) = trimStationOverlaps(W, W_end, W_size, voxel_data_f)

    # Fuse stations to single volumes
    volume_w = fuseVolume(W, W_end, W_size, voxel_data_w, overlaps) 
    volume_f = fuseVolume(W, W_end, W_size, voxel_data_f, overlaps)

    # Compress to two-dimensional representations
    mip_out = formatOutput(volume_w, volume_f)

    return (mip_out, True)


# Check if water and fat stations are actually in the same position, non-redundant, and of the same size.
# The latter is not always true, in rare cases even the spacing does not seem to match
def ensureStationConsistency(voxel_data_w, voxel_data_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings):

    # Abort if water and fat stations are not in the same positions
    if not np.allclose(positions_w, positions_f):
        print("        ABORT: Water and fat stations are not in the same position!")
        return (False, voxel_data_w, voxel_data_f, positions_w)

    # In case of redundant stations, choose the newest
    if len(np.unique(positions_w, axis=0)) != len(positions_w):

        idx = []

        for pos in np.unique(positions_w, axis=0):

            # Find stations at current position
            offsets = np.array(positions_w) - np.tile(pos, (len(positions_w), 1))
            dist = np.sum(np.abs(offsets), axis=1)

            indices_p = np.where(dist == 0)[0]

            if len(indices_p) > 1:

                # Choose newest station
                timestamps_w_p = [str(x).replace(".", "") for f, x in enumerate(timestamps_w) if f in indices_p]

                # If you get scanned around midnight its your own fault
                recent_p = np.argmax(np.array(timestamps_w_p))
                print("        WARNING: Image stations ({}) are superimposed. Choosing most recently imaged one ({})".format(indices_p, indices_p[recent_p]))
                
                idx.append(indices_p[recent_p])
            else:
                idx.append(indices_p[0])
        
        # Extract selected stations
        idx = np.array(idx)

        #
        voxel_data_w = voxel_data_w[idx]
        positions_w = positions_w[idx]
        timestamps_w = timestamps_w[idx]

        #
        voxel_data_f = voxel_data_f[idx]
        positions_f = positions_f[idx]
        timestamps_f = timestamps_f[idx]

        pixel_spacings = pixel_spacings[idx]

    # Crop corresponding stations to same size where necessary
    for i in range(len(positions_w)):

        if not np.array_equal(voxel_data_w[i].shape, voxel_data_f[i].shape):

            print("        WARNING: Corresponding stations {} have different dimensions: {} vs {} (Water vs Fat)".format(i, voxel_data_w[i].shape, voxel_data_f[i].shape))
            print("                 Cutting to largest common size")
            # Cut to common size
            min_size = np.amin(np.vstack((voxel_data_w[i].shape, voxel_data_f[i].shape)), axis=0)

            voxel_data_w[i] = np.ascontiguousarray(voxel_data_w[i][:min_size[0], :min_size[1], :min_size[2]])
            voxel_data_f[i] = np.ascontiguousarray(voxel_data_f[i][:min_size[0], :min_size[1], :min_size[2]])

    # Sort stations by position
    pos_z = np.array(positions_w)[:, 2]
    idx = np.argsort(pos_z)[::-1]

    #
    voxel_data_w = voxel_data_w[idx]
    positions_w = positions_w[idx]
    timestamps_w = timestamps_w[idx]

    #
    voxel_data_f = voxel_data_f[idx]
    positions_f = positions_f[idx]
    timestamps_f = timestamps_f[idx]

    pixel_spacings = pixel_spacings[idx]

    return (True, voxel_data_w, voxel_data_f, positions_w, pixel_spacings)


# Create mean intensity projections (MIPs) of water and fat signal along coronal and sagittal view
# along with coronal and sagittal fat fraction slice
def formatOutput(volume_w, volume_f):

    (volume_ff, mask) = calculateFractions(volume_w, volume_f)

    # Create mean intensity projections (MIP) with fat fraction slice
    mip_w = formatMip(volume_w)
    mip_f = formatMip(volume_f)
    ff = formatFF(volume_ff, mask)

    mip_out = np.dstack((mip_w, mip_f, ff)) # original implementation
    #mip_out = np.dstack((ff, mip_f, mip_w)) # visualization used in paper

    mip_out = mip_out.transpose(2, 0, 1)

    return mip_out


def calculateFractions(volume_w, volume_f):

    # Create sum image
    volume_sum = volume_w + volume_f
    volume_sum[volume_sum == 0] = 1

    # Calculate fraction images
    #volume_wf = 1000 * volume_w / volume_sum
    volume_ff = 1000 * volume_f / volume_sum

    # Calculate threshold for body mask
    # based on average otsu threshold of all slices
    ts = np.zeros(volume_sum.shape[1])
    for i in range(volume_sum.shape[1]):
        ts[i] = filters.threshold_otsu(volume_sum[:, i, :])

    t = np.mean(ts)

    # Create mask
    volume_mask = np.ones(volume_w.shape).astype("bool")
    volume_mask[volume_sum < t] = 0

    return (volume_ff, volume_mask)


def fuseVolume(W, W_end, W_size, voxel_data, overlaps):

    S = len(voxel_data)

    # Cast to datatype
    for i in range(S):  
        voxel_data[i] = voxel_data[i].astype(c_datatype_numpy)

    # Taper off station edges linearly for later addition
    if c_interpolate_seams:
        voxel_data = fadeStationEdges(overlaps, W_size, voxel_data)

    # Adjust mean intensity of overlapping slices
    if c_correct_intensity:
        voxel_data = correctOverlapIntensity(overlaps, W_size, voxel_data)

    # Combine stations into volume by addition
    volume = combineStationsToVolume(W, W_end, voxel_data)

    # Remove slices affected by folding
    if c_trim_axial_slices > 0:
        start = c_trim_axial_slices
        end = volume.shape[2] - c_trim_axial_slices
        volume = volume[:, :, start:end]

    return volume


# Locate center or quarter of body mass along axis
def getSliceOfMass(mass, mask, axis):

    com_i = 0

    shifts = np.array(mask.shape)

    for i in range(mask.shape[axis]):
        shifts[axis] = i
        mass_i = np.count_nonzero(mask[:shifts[0], :shifts[1], :shifts[2]])
        if mass_i >= mass:
            com_i = i
            break

    return com_i


# Encode fat fraction values to 8 bit
def formatFractionSlice(img):

    img = np.rot90(img, 1)
    img = np.clip(img / 500., 0, 1) * 255 # Encode percentages of 0-50%
    img = img.astype("uint8")

    return img


# Format fat fraction slices that should cut through the liver from
# center of mass in coronal view
# quarter of mass in sagittal view
def formatFF(volume, mask):

    # Crop MRI bed
    bed_width = 22
    volume = volume[:, :volume.shape[1]-bed_width, :]
    mask = mask[:, :mask.shape[1]-bed_width, :]

    # Determine centers of mass
    mass = np.count_nonzero(mask)
    mass_sag_half = np.count_nonzero(mask[:int(mask.shape[0] / 2), :, :])
    
    # Get coronal slice
    com_cor = getSliceOfMass(mass / 2, mask, 1)
    slice_cor = formatFractionSlice(volume[:, com_cor, :])

    # Get sagittal slice
    com_sag = getSliceOfMass(mass_sag_half / 2, mask, 0)
    slice_sag = formatFractionSlice(volume[com_sag, :, :])

    # Combine to single output
    slice_out = np.concatenate((slice_cor, slice_sag), 1)
    slice_out = cv2.resize(slice_out, (256, 256))

    return slice_out


# Generate mean intensity projection 
def formatMip(volume):

    # Crop MRI bed
    bed_width = 22
    volume = volume[:, :volume.shape[1]-bed_width, :]

    # Coronal projection
    slice_cor = np.sum(volume, axis = 1)
    slice_cor = np.rot90(slice_cor, 1)

    # Sagittal projection
    slice_sag = np.sum(volume, axis = 0)
    slice_sag = np.rot90(slice_sag, 1)

    # Normalize intensities
    slice_cor = (normalize(slice_cor) * 255).astype("uint8")
    slice_sag = (normalize(slice_sag) * 255).astype("uint8")

    # Combine to single output
    slice_out = np.concatenate((slice_cor, slice_sag), 1)
    slice_out = cv2.resize(slice_out, (256, 256))

    return slice_out


def normalize(img):

    img = img.astype("float")
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return img


# Write stations to common voxel grid
def combineStationsToVolume(W, W_end, voxel_data):

    S = len(voxel_data)

    volume_dim = np.amax(W_end, axis=0).astype("int")
    volume = np.zeros(volume_dim)

    for i in range(S):
        volume[W[i, 0]:W_end[i, 0], W[i, 1]:W_end[i, 1], W[i, 2]:W_end[i, 2]] += voxel_data[i][:, :, :]

    #
    volume = np.flip(volume, 2)
    volume = np.swapaxes(volume, 0, 1)

    return volume


# Select data of all stations with given tag in name
def extractStationsWithTag(tag, station_names, station_voxel_data, station_positions, station_pixel_spacings, station_timestamps):

    idx = [f for f, x in enumerate(station_names) if str(tag) in str(x)]
    idx = np.array(idx)

    voxel_data_t = station_voxel_data[idx]
    positions_t = station_positions[idx]
    pixel_spacings_t = station_pixel_spacings[idx]
    timestamps_t = station_timestamps[idx]
    
    return (voxel_data_t, positions_t, pixel_spacings_t, timestamps_t)


def getSignalSliceNamesInZip(z):

    file_names = [f.filename for f in z.infolist()]

    # Search for manifest file (name may be misspelled)
    csv_name = [f for f in file_names if "manifest" in f][0]

    with z.open(csv_name) as f0:

        data = f0.read() # Decompress into memory

        entries = str(data).split("\\n")
        entries.pop(-1)

        # Remove trailing blank lines
        entries = [f for f in entries if f != ""]

        # Get indices of relevant columns
        header_elements = entries[0].split(",")
        column_filename = [f for f,x in enumerate(header_elements) if "filename" in x][0]

        # Search for tags such as "Dixon_noBH_F". The manifest header can not be relied on
        for e in entries:
            entry_parts = e.split(",")
            column_desc = [f for f,x in enumerate(entry_parts) if "Dixon_noBH_F" in x]

            if column_desc:
                column_desc = column_desc[0]
                break

        # Get slice descriptions and filenames
        descriptions = [f.split(",")[column_desc] for f in entries]
        filenames = [f.split(",")[column_filename] for f in entries]

        # Extract signal images only
        chosen_rows = [f for f,x in enumerate(descriptions) if "_W" in x or "_F" in x]
        chosen_filenames = [x for f,x in enumerate(filenames) if f in chosen_rows]

    return chosen_filenames


##
# Return, for S stations:
# R:     station start coordinates, shape Sx3
# R_end: station end coordinates,   shape Sx3
# dims:  station extents,           shape Sx3
# 
# Coordinates in R and R_end are in the voxel space of the first station
def getReadCoordinates(voxel_data, positions, pixel_spacings):

    S = len(voxel_data)

    # Convert from list to arrays
    positions = np.array(positions)
    pixel_spacings = np.array(pixel_spacings)

    # Get dimensions of stations
    dims = np.zeros((S, 3))
    for i in range(S):
        dims[i, :] = voxel_data[i].shape

    # Get station start coordinates
    R = positions
    origin = np.array(R[0])
    for i in range(S):
        R[i, :] = (R[i, :] - origin) / c_out_pixel_spacing

    R[:, 0] -= np.amin(R[:, 0])
    R[:, 1] -= np.amin(R[:, 1])
    R[:, 2] *= -1

    R[:, [0, 1]] = R[:, [1, 0]]

    # Get station end coordinates
    R_end = np.array(R)
    for i in range(S):
        R_end[i, :] += dims[i, :] * pixel_spacings[i, :] / c_out_pixel_spacing

    return (R, R_end, dims)


##
# Linearly taper off voxel values along overlap of two stations, 
# so that their addition leads to a linear interpolation.
def fadeStationEdges(overlaps, W_size, voxel_data):

    S = len(voxel_data)

    for i in range(S):

        # Only fade inwards facing edges for outer stations
        fadeToPrev = (i > 0)
        fadeToNext = (i < (S - 1))

        # Fade ending edge (facing to next station)
        if fadeToNext:

            for j in range(overlaps[i]):
                factor = (j+1) / (float(overlaps[i]) + 1) # exclude 0 and 1
                voxel_data[i][:, :, W_size[i, 2] - 1 - j] *= factor

        # Fade starting edge (facing to previous station)
        if fadeToPrev:

            for j in range(overlaps[i-1]):
                factor = (j+1) / (float(overlaps[i-1]) + 1) # exclude 0 and 1
                voxel_data[i][:, :, j] *= factor

    return voxel_data


## 
# Take mean intensity of slices at the edge of the overlap between stations i and (i+1)
# Adjust mean intensity of each slice along the overlap to linear gradient between these means
def correctOverlapIntensity(overlaps, W_size, voxel_data):

    S = len(voxel_data)

    for i in range(S - 1):
        overlap = overlaps[i]

        # Get average intensity at outer ends of overlap
        edge_a = voxel_data[i+1][:, :, overlap]
        edge_b = voxel_data[i][:, :, W_size[i, 2] - 1 - overlap]

        mean_a = np.mean(edge_a)
        mean_b = np.mean(edge_b)

        for j in range(overlap):

            # Get desired mean intensity along gradient
            factor = (j+1) / (float(overlap) + 1)
            target_mean = mean_b + (mean_a - mean_b) * factor

            # Get current mean of slice when both stations are summed
            slice_b = voxel_data[i][:, :, W_size[i, 2] - overlap + j]
            slice_a = voxel_data[i+1][:, :, j]

            slice_mean = np.mean(slice_a) + np.mean(slice_b)

            # Get correction factor
            correct = target_mean / slice_mean

            voxel_data[i][:, :, W_size[i, 2] - overlap + j] *= correct
            voxel_data[i+1][:, :, j] *= correct

    return voxel_data


##
# Ensure that the stations i and (i + 1) overlap by at most c_max_overlap.
# Trim any excess symmetrically
# Update their extents in W and W_end
def trimStationOverlaps(W, W_end, W_size, voxel_data):

    W = np.array(W)
    W_end = np.array(W_end)
    W_size = np.array(W_size)

    S = len(voxel_data)
    overlaps = np.zeros(S).astype("int")

    for i in range(S - 1):
        # Get overlap between current and next station
        overlap = W_end[i, 2] - W[i + 1, 2]

        # No overlap
        if overlap <= 0:
            print("        WARNING: No overlap between stations {} and {}. Image might be faulty.".format(i, i+1))

        # Small overlap which can for interpolation
        elif overlap <= c_max_overlap and c_interpolate_seams:
            print("        WARNING: Overlap between stations {} and {} is only {}. Using this overlap for interpolation".format(i, i+1, overlap))

        # Large overlap which must be cut
        else:
            if c_interpolate_seams:
                # Keep an overlap of at most c_max_overlap
                cut_a = (overlap - c_max_overlap) / 2.
                overlap = c_max_overlap
            else:
                # Cut at center of seam
                cut_a = overlap / 2.
                overlap = 0

            cut_b = int(np.ceil(cut_a))
            cut_a = int(np.floor(cut_a))

            voxel_data[i] = voxel_data[i][:, :, 0:(W_size[i, 2] - cut_a)]
            voxel_data[i + 1] = voxel_data[i + 1][:, :, cut_b:]

            #
            W_end[i, 2] = W_end[i, 2] - cut_a
            W_size[i, 2] -= cut_a

            W[i + 1, 2] = W[i + 1, 2] + cut_b
            W_size[i + 1, 2] -= cut_b

        overlaps[i] = overlap

    return (overlaps, W, W_end, W_size, voxel_data)


##
# Station voxels are positioned at R to R_end, not necessarily aligned with output voxel grid
# Resample stations onto voxel grid of output volume
def resampleStations(voxel_data, positions, pixel_spacings):

    # R: station positions off grid respective to output volume
    # W: station positions on grid after resampling
    (R, R_end, dims) = getReadCoordinates(voxel_data, positions, pixel_spacings)

    # Get coordinates of voxels to write to
    W = np.around(R).astype("int")
    W_end = np.around(R_end).astype("int")
    W_size = W_end - W

    result_data = []

    #
    for i in range(len(voxel_data)):

        # Get largest offset off of voxel grid
        offsets = np.concatenate((R[i, :].flatten(), R_end[i, :].flatten()))
        offsets = np.abs(offsets - np.around(offsets))

        max_offset = np.amax(offsets)

        # Get difference in voxel counts
        voxel_count_out = np.around(W_size[i, :])
        voxel_count_dif = np.sum(voxel_count_out - dims[i, :])

        # No resampling if station voxels are already aligned with output voxel grid
        doResample = (max_offset > c_resample_tolerance or voxel_count_dif != 0)

        result = None
        
        if doResample:

            # Use numba implementation on gpu:
            scalings = (R_end[i, :] - R[i, :]) / dims[i, :]
            offsets = R[i, :] - W[i, :] 

            result = numba_interpolate.interpolate3d(W_size[i, :], voxel_data[i], scalings, offsets)

        else:
            # No resampling necessary
            result = voxel_data[i]

        result_data.append(result.reshape(W_size[i, :]))

    return (result_data, W, W_end, W_size)


def groupSlicesToStations(slice_pixel_data, slice_series, slice_names, slice_positions, slice_pixel_spacings, slice_times):

    # Group by series into stations
    unique_series = np.unique(slice_series)

    #
    station_voxel_data = []
    station_series = []
    station_names = []
    station_positions = []
    station_voxel_spacings = []
    station_times = []

    # Each series forms one station
    for s in unique_series:

        # Get slice indices for series s
        indices_s = [f for f, x in enumerate(slice_series) if x == s]

        # Get physical positions of slices
        slice_positions_s = [x for f, x in enumerate(slice_positions) if f in indices_s]

        position_max = np.amax(np.array(slice_positions_s).astype("float"), axis=0)
        station_positions.append(position_max)

        # Combine slices to station
        voxel_data_s = slicesToStationData(indices_s, slice_positions_s, slice_pixel_data)
        station_voxel_data.append(voxel_data_s)

        # Get index of first slice
        slice_0 = indices_s[0]

        station_series.append(slice_series[slice_0])
        station_names.append(slice_names[slice_0])
        station_times.append(slice_times[slice_0])

        # Get 3d voxel spacing
        voxel_spacing_2d = slice_pixel_spacings[slice_0]

        # Get third dimension by dividing station extent by slice count
        z_min = np.amin(np.array(slice_positions_s)[:, 2].astype("float"))
        z_max = np.amax(np.array(slice_positions_s)[:, 2].astype("float"))
        z_spacing = (z_max - z_min) / (len(slice_positions_s) - 1)

        voxel_spacing = np.hstack((voxel_spacing_2d, z_spacing))
        station_voxel_spacings.append(voxel_spacing)

    return (station_voxel_data, station_names, station_positions, station_voxel_spacings, station_times)


def getDataFromDicom(ds):

    pixel_data = np.array(ds.pixel_array).astype("float32")

    series = ds.get_item(["0020", "0011"]).value
    series = int(series)

    name = ds.get_item(["0008", "103e"]).value

    position = ds.get_item(["0020", "0032"]).value 
    position = np.array(position.decode().split("\\")).astype("float32")

    pixel_spacing = ds.get_item(["0028", "0030"]).value
    pixel_spacing = np.array(pixel_spacing.decode().split("\\")).astype("float32")

    start_time = ds.get_item(["0008", "0031"]).value

    return (pixel_data, series, name, position, pixel_spacing, start_time)


def slicesToStationData(slice_indices, slice_positions, slices):

    # Get size of output volume station
    slice_count = len(slice_indices)
    slice_shape = slices[slice_indices[0]].shape

    # Get slice positions
    slices_z = np.zeros(slice_count)
    for z in range(slice_count):
        slices_z[z] = slice_positions[z][2]

    # Sort slices by position
    idx = np.argsort(slices_z)[::-1]
    slices_z = np.array(slices_z)[idx]
    slice_indices = np.array(slice_indices)[idx]

    # Write slices to volume station
    dim = np.array((slice_shape[0], slice_shape[1], slice_count))
    station = np.zeros(dim)

    for z in range(dim[2]):
        slice_z_index = slice_indices[z]
        station[:, :, z] = slices[slice_z_index]

    return station


def stationsFromDicom(input_path_zip):

    # Get slice info
    pixel_data = []
    series = []
    names = []
    positions = []
    pixel_spacings = []
    times = []

    time_x = time.time()

    #
    z = zipfile.ZipFile(input_path_zip)

    signal_slice_names = getSignalSliceNamesInZip(z)

    for i in range(len(signal_slice_names)):

        # Read signal slices in memory
        with z.open(signal_slice_names[i]) as f0:

            data = f0.read() # Decompress into memory
            ds = pydicom.read_file(io.BytesIO(data)) # Read from byte stream

            (pixel_data_i, series_i, name_i, position_i, spacing_i, time_i) = getDataFromDicom(ds)

            pixel_data.append(pixel_data_i)
            series.append(series_i)
            names.append(name_i)
            positions.append(position_i)
            pixel_spacings.append(spacing_i)
            times.append(time_i)

    z.close()

    (station_voxel_data, station_names, station_positions, station_voxel_spacings, station_times) = groupSlicesToStations(pixel_data, series, names, positions, pixel_spacings, times)

    station_voxel_data = np.array(station_voxel_data)
    station_names = np.array(station_names)
    station_positions = np.array(station_positions)
    station_voxel_spacings = np.array(station_voxel_spacings)
    station_times = np.array(station_times)

    return (station_voxel_data, station_names, station_positions, station_voxel_spacings, station_times)


class DicomDataset(data.Dataset):

    def __init__(self, dicom_paths):
        self.dicom_paths = dicom_paths

    def __len__(self):
        return len(self.dicom_paths)


    def __getitem__(self, index):

        dicom_path = self.dicom_paths[index]
        (mip_out, success) = fuseAndProjectDicom(dicom_path)

        if not success:
            mip_out = np.zeros((256, 256, 3))

        return (mip_out, dicom_path, not success)

import os
import sys
import io

import time

import zipfile
import pydicom

import numpy as np

import scipy.interpolate
import numba_interpolate

import skimage.measure 
from skimage import filters

import cv2
import nrrd

c_out_pixel_spacing = np.array((2.23214293, 2.23214293, 3.))
c_resample_tolerance = 0.01 # Only interpolate voxels further off of the voxel grid than this

c_interpolate_seams = True # If yes, cut overlaps between segments to at most c_max_overlap and interpolate along them, otherwise cut at center of overlap
c_correct_intensity = True # If yes, apply intensity correction along overlap
c_max_overlap = 8 # Used in interpolation, any segment overlaps are cut to be most this many voxels in size

c_trim_axial_slices = 4 # Trim this many axial slices from the output volume to remove folding artefacts

c_store_signals = True # If yes, store signal images

c_store_fractions = False # If yes, calculate fat and water fraction by segment and fuse the result. The resulting images can not necessarily be calculated from the signal images directly
c_mask_fractions = False # If yes, attempt to remove background noise from the fraction images
c_mask_ratio = 0.1 # When creating fraction images, mask out voxels darker than this ratio of the total range of intensities

c_store_nrrd = False
c_store_mip = True
c_mip_encode_fraction = True # When writing mips, normalize the water or fat fractions

c_datatype_numpy = "float32" # See: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
c_datatype_nrrd = "float"    # See: https://github.com/mhe/pynrrd/blob/master/nrrd/reader.py

c_use_gpu = True # If yes, use numba for gpu access, otherwise use scipy on cpu


#def dicomToVolume(input_path_zip, output_path, version_tag):
def compressToMip(input_path_zip, output_path):

    if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path))

    if not os.path.exists(input_path_zip):
        print("    Could not find input file {}".format(input_path_zip))
        return

    (seg_voxel_data, seg_names, seg_positions, seg_pixel_spacings, seg_timestamps) = segmentsFromDicom(input_path_zip)

    origin = np.amin(np.array(seg_positions), axis=0)

    # Find water and fat signal segment data
    (voxel_data_w, positions_w, pixel_spacings, timestamps_w) = extractSegmentsForModality("_W", seg_names, seg_voxel_data, seg_positions, seg_pixel_spacings, seg_timestamps)
    (voxel_data_f, positions_f, _, timestamps_f)              = extractSegmentsForModality("_F", seg_names, seg_voxel_data, seg_positions, seg_pixel_spacings, seg_timestamps)

    # Ensure that water and fat segments match in position and size and non-redundant
    (segments_consistent, voxel_data_w, voxel_data_f, positions, pixel_spacings) = ensureSegmentConsistency(voxel_data_w, voxel_data_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings)
    if not segments_consistent: return

    # Resample segments onto output volume voxel grid
    (voxel_data_w, _, _, _)          = resampleSegments(voxel_data_w, positions, pixel_spacings)
    (voxel_data_f, W, W_end, W_size) = resampleSegments(voxel_data_f, positions, pixel_spacings)

    # Cut segment overlaps to at most c_max_overlap
    (_, _, _, _, voxel_data_w)                 = trimSegmentOverlaps(W, W_end, W_size, voxel_data_w)
    (overlaps, W, W_end, W_size, voxel_data_f) = trimSegmentOverlaps(W, W_end, W_size, voxel_data_f)

    #
    volume_w = fuseVolume(W, W_end, W_size, voxel_data_w, overlaps) 
    volume_f = fuseVolume(W, W_end, W_size, voxel_data_f, overlaps)

    #
    storeOutput(volume_w, volume_f, output_path, origin)


def ensureSegmentConsistency(voxel_data_w, voxel_data_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings):

    # Abort if water and fat segments are not in the same positions
    if not np.allclose(positions_w, positions_f):
        print("ABORT: Water and fat segments are not in the same position!")
        return (False, voxel_data_w, voxel_data_f, positions_w)

    # In case of redundant segments, choose the newest
    if len(np.unique(positions_w, axis=0)) != len(positions_w):

        seg_select = []

        for pos in np.unique(positions_w, axis=0):

            # Find segments at current position
            offsets = np.array(positions_w) - np.tile(pos, (len(positions_w), 1))
            dist = np.sum(np.abs(offsets), axis=1)

            indices_p = np.where(dist == 0)[0]

            if len(indices_p) > 1:

                # Choose newest segment
                timestamps_w_p = [str(x).replace(".", "") for f, x in enumerate(timestamps_w) if f in indices_p]

                # If you get scanned around midnight its your own fault
                recent_p = np.argmax(np.array(timestamps_w_p))

                print("WARNING: Image segments ({}) are superimposed. Choosing most recently imaged one ({})".format(indices_p, indices_p[recent_p]))
                
                seg_select.append(indices_p[recent_p])
            else:
                seg_select.append(indices_p[0])
        
        voxel_data_w = [x for f,x in enumerate(voxel_data_w) if f in seg_select]        
        positions_w = [x for f,x in enumerate(positions_w) if f in seg_select]        
        timestamps_w = [x for f,x in enumerate(timestamps_w) if f in seg_select]        

        voxel_data_f = [x for f,x in enumerate(voxel_data_f) if f in seg_select]        
        positions_f = [x for f,x in enumerate(positions_f) if f in seg_select]        
        timestamps_f = [x for f,x in enumerate(timestamps_f) if f in seg_select]        

        pixel_spacings = [x for f,x in enumerate(pixel_spacings) if f in seg_select]        

    # Crop corresponding segments to same size where necessary
    for i in range(len(positions_w)):

        if not np.array_equal(voxel_data_w[i].shape, voxel_data_f[i].shape):

            print("WARNING: Corresponding segments {} have different dimensions: {} vs {} (Water vs Fat)".format(i, voxel_data_w[i].shape, voxel_data_f[i].shape))
            print("         Cutting to largest common size")
            # Cut to common size
            min_size = np.amin(np.vstack((voxel_data_w[i].shape, voxel_data_f[i].shape)), axis=0)

            voxel_data_w[i] = np.ascontiguousarray(voxel_data_w[i][:min_size[0], :min_size[1], :min_size[2]])
            voxel_data_f[i] = np.ascontiguousarray(voxel_data_f[i][:min_size[0], :min_size[1], :min_size[2]])

    # Sort by position
    pos_z = np.array(positions_w)[:, 2]
    (pos_z, pos_indices) = zip(*sorted(zip(pos_z, np.arange(len(pos_z))), reverse=True))

    voxel_data_w = [voxel_data_w[i] for i in pos_indices]
    positions_w = [positions_w[i] for i in pos_indices]
    timestamps_w = [timestamps_w[i] for i in pos_indices]

    voxel_data_f = [voxel_data_f[i] for i in pos_indices]
    positions_f = [positions_f[i] for i in pos_indices]
    timestamps_f = [timestamps_f[i] for i in pos_indices]

    pixel_spacings = [pixel_spacings[i] for i in pos_indices]

    return (True, voxel_data_w, voxel_data_f, positions_w, pixel_spacings)


def storeOutput(volume_w, volume_f, output_path, origin):

    (volume_wf, volume_ff, mask) = calculateFractions(volume_w, volume_f)

    # Create mean intensity projections (MIP) with fat fraction slice
    mip_w = formatMip(volume_w)
    mip_f = formatMip(volume_f)
    ff = formatFF(volume_ff, mask)

    mip_out = np.dstack((mip_w, mip_f, ff)) # original implementation
    #mip_out = np.dstack((ff, mip_f, mip_w)) # visualization used in paper

    #cv2.imwrite(output_path + ".png", mip_out)
    np.save(output_path, mip_out.transpose(2, 0, 1))


def calculateFractions(volume_w, volume_f):

    volume_sum = volume_w + volume_f
    volume_sum[volume_sum == 0] = 1

    volume_wf = 1000 * volume_w / volume_sum
    volume_ff = 1000 * volume_f / volume_sum


    # Mask fraction images
    #t = np.amin(sum) + c_mask_ratio * (np.amax(sum) - np.amin(sum))
    #volume_mask = np.ones(volume_w.shape).astype("uint8")
    #volume_mask[sum < t] = 0

    #bed_width = 22
    #bed_max = np.max(volume_sum[:, volume_sum.shape[1]-bed_width:, :])
    #volume_mask = np.ones(volume_w.shape).astype("uint8")
    #volume_mask[volume_sum < bed_max] = 0

    ts = np.zeros(volume_sum.shape[1])
    for i in range(volume_sum.shape[1]):
        ts[i] = filters.threshold_otsu(volume_sum[:, i, :])

    t = np.mean(ts)

    volume_mask = np.ones(volume_w.shape).astype("uint8")
    volume_mask[volume_sum < t] = 0

    # Get connected components to isolate background only (slow)
    #labels = skimage.measure.label(volume_mask) 
    #L = len(np.unique(labels))
    #label_intensities = np.zeros(L)
    #for i in range(L):
    #    l = np.unique(labels)[i]
    #    label_intensities[i] = np.sum(1 - volume_mask[labels == l])

    #max_label = np.unique(labels)[np.argmax(label_intensities)]
    #foreground = labels != max_label

    #end_time = time.time()

    #print("Time for connected component search when masking fraction images: {}".format(end_time - start_time))

    #volume_wf = np.multiply(volume_wf, volume_mask)
    #volume_ff = np.multiply(volume_ff, volume_mask)

    return (volume_wf, volume_ff, volume_mask)


def fuseVolume(W, W_end, W_size, voxel_data, overlaps):

    S = len(voxel_data)

    # Cast to datatype
    for i in range(S):  
        voxel_data[i] = voxel_data[i].astype(c_datatype_numpy)

    # Taper off segment edges linearly for later addition
    if c_interpolate_seams:
        voxel_data = fadeSegmentEdges(overlaps, W_size, voxel_data)

    # Adjust mean intensity of overlapping slices
    if c_correct_intensity:
        voxel_data = correctOverlapIntensity(overlaps, W_size, voxel_data)

    # Combine segments into volume by addition
    volume = combineSegmentsToVolume(W, W_end, voxel_data)

    # Remove slices affected by folding
    if c_trim_axial_slices > 0:
        start = c_trim_axial_slices
        end = volume.shape[2] - c_trim_axial_slices
        volume = volume[:, :, start:end]

    return volume


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


def formatFractionSlice(img):

    img = np.rot90(img, 1)
    #img = np.clip(img / 1000., 0, 1) * 255
    img = np.clip(img / 500., 0, 1) * 255 # Encode percentages of 0-50%
    img = img.astype("uint8")

    return img


def writeFF(volume, mask, out_path):

    bed_width = 22
    volume = volume[:, :volume.shape[1]-bed_width, :]
    mask = mask[:, :mask.shape[1]-bed_width, :]

    # Determine centers of mass
    mass = np.count_nonzero(mask)
    mass_sag_half = np.count_nonzero(mask[:int(mask.shape[0] / 2), :, :])
    
    #
    com_cor = getSliceOfMass(mass / 2, mask, 1)
    slice_cor = formatFractionSlice(volume[:, com_cor, :])

    com_sag = getSliceOfMass(mass_sag_half / 2, mask, 0)
    slice_sag = formatFractionSlice(volume[com_sag, :, :])

    # Combine to single output
    slice_out = np.concatenate((slice_cor, slice_sag), 1)

    slice_out = slice_out[:176, :]
    slice_out = cv2.resize(slice_out, (376, 176))
    #slice_out = slice_out[:128, :]

    #cv2.imwrite(out_path + "_proj.png", slice_out)
    #slice_out = np.swapaxes(np.swapaxes(slice_out, 0, 1), 0, 2)
    #slice_out = np.swapaxes(np.swapaxes(slice_out, 0, 1), 0, 2)
    #print(slice_out.shape)
    slice_out = slice_out.reshape(1, 176, 376)

    np.save(out_path + ".npy", slice_out)


def formatFF(volume, mask):

    bed_width = 22
    volume = volume[:, :volume.shape[1]-bed_width, :]
    mask = mask[:, :mask.shape[1]-bed_width, :]

    # Determine centers of mass
    mass = np.count_nonzero(mask)
    mass_sag_half = np.count_nonzero(mask[:int(mask.shape[0] / 2), :, :])
    
    #
    com_cor = getSliceOfMass(mass / 2, mask, 1)
    slice_cor = formatFractionSlice(volume[:, com_cor, :])

    com_sag = getSliceOfMass(mass_sag_half / 2, mask, 0)
    slice_sag = formatFractionSlice(volume[com_sag, :, :])

    # Clip top 1% (no benefit for networks)
    #slice_cor = np.clip(slice_cor, np.amin(slice_cor), np.percentile(slice_cor, 99))
    #slice_sag = np.clip(slice_sag, np.amin(slice_sag), np.percentile(slice_sag, 99))

    # Combine to single output
    slice_out = np.concatenate((slice_cor, slice_sag), 1)

    #slice_out = slice_out[:176, :]
    #slice_out = cv2.resize(slice_out, (376, 176))
    #slice_out = slice_out.reshape(1, 176, 376)
    slice_out = cv2.resize(slice_out, (256, 256))

    return slice_out



# Generate mean intensity projection 
def formatMip(volume):

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


def combineSegmentsToVolume(W, W_end, voxel_data):

    S = len(voxel_data)

    volume_dim = np.amax(W_end, axis=0).astype("int")
    volume = np.zeros(volume_dim)

    for i in range(S):
        volume[W[i, 0]:W_end[i, 0], W[i, 1]:W_end[i, 1], W[i, 2]:W_end[i, 2]] += voxel_data[i][:, :, :]

    #
    volume = np.flip(volume, 2)
    volume = np.swapaxes(volume, 0, 1)

    return volume


def extractSegmentsForModality(tag, segment_names, segment_voxel_data, segment_positions, segment_pixel_spacings, segment_timestamps):

    # Merge all segments with given tag
    indices_t = [f for f, x in enumerate(segment_names) if str(tag) in str(x)]

    voxel_data_t = [x for f, x in enumerate(segment_voxel_data) if f in indices_t]
    positions_t = [x for f, x in enumerate(segment_positions) if f in indices_t]
    pixel_spacings_t = [x for f, x in enumerate(segment_pixel_spacings) if f in indices_t]
    timestamps_t = [x for f, x in enumerate(segment_timestamps) if f in indices_t]
    
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
# Return, for S segments:
# R:     segment start coordinates, shape Sx3
# R_end: segment end coordinates,   shape Sx3
# dims:  segment extents,           shape Sx3
# 
# Coordinates in R and R_end are in the voxel space of the first segment
def getReadCoordinates(voxel_data, positions, pixel_spacings):

    S = len(voxel_data)

    # Convert from list to arrays
    positions = np.array(positions)
    pixel_spacings = np.array(pixel_spacings)

    # Get dimensions of segments
    dims = np.zeros((S, 3))
    for i in range(S):
        dims[i, :] = voxel_data[i].shape

    # Get segment start coordinates
    R = positions
    origin = np.array(R[0])
    for i in range(S):
        R[i, :] = (R[i, :] - origin) / c_out_pixel_spacing

    R[:, 0] -= np.amin(R[:, 0])
    R[:, 1] -= np.amin(R[:, 1])
    R[:, 2] *= -1

    R[:, [0, 1]] = R[:, [1, 0]]

    # Get segment end coordinates
    R_end = np.array(R)
    for i in range(S):
        R_end[i, :] += dims[i, :] * pixel_spacings[i, :] / c_out_pixel_spacing

    return (R, R_end, dims)


##
# Linearly taper off voxel values along overlap of two segments, 
# so that their addition leads to a linear interpolation.
def fadeSegmentEdges(overlaps, W_size, voxel_data):

    S = len(voxel_data)

    for i in range(S):

        # Only fade inwards facing edges for outer segments
        fadeToPrev = (i > 0)
        fadeToNext = (i < (S - 1))

        # Fade ending edge (facing to next segment)
        if fadeToNext:

            for j in range(overlaps[i]):
                factor = (j+1) / (float(overlaps[i]) + 1) # exclude 0 and 1
                voxel_data[i][:, :, W_size[i, 2] - 1 - j] *= factor

        # Fade starting edge (facing to previous segment)
        if fadeToPrev:

            for j in range(overlaps[i-1]):
                factor = (j+1) / (float(overlaps[i-1]) + 1) # exclude 0 and 1
                voxel_data[i][:, :, j] *= factor

    return voxel_data


## 
# Take mean intensity of slices at the edge of the overlap between segments i and (i+1)
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

            # Get current mean of slice when both segments are summed
            slice_b = voxel_data[i][:, :, W_size[i, 2] - overlap + j]
            slice_a = voxel_data[i+1][:, :, j]

            slice_mean = np.mean(slice_a) + np.mean(slice_b)

            # Get correction factor
            correct = target_mean / slice_mean

            voxel_data[i][:, :, W_size[i, 2] - overlap + j] *= correct
            voxel_data[i+1][:, :, j] *= correct

    return voxel_data


##
# Ensure that the segments i and (i + 1) overlap by at most c_max_overlap.
# Trim any excess symmetrically
# Update their extents in W and W_end
def trimSegmentOverlaps(W, W_end, W_size, voxel_data):

    W = np.array(W)
    W_end = np.array(W_end)
    W_size = np.array(W_size)

    S = len(voxel_data)
    overlaps = np.zeros(S).astype("int")

    for i in range(S - 1):
        # Get overlap between current and next segment
        overlap = W_end[i, 2] - W[i + 1, 2]

        # No overlap
        if overlap <= 0:
            print("WARNING: No overlap between segments {} and {}. Image might be faulty.".format(i, i+1))

        # Small overlap which can for interpolation
        elif overlap <= c_max_overlap and c_interpolate_seams:
            print("WARNING: Overlap between segments {} and {} is only {}. Using this overlap for interpolation".format(i, i+1, overlap))

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
# Segment voxels are positioned at R to R_end, not necessarily aligned with output voxel grid
# Resample segments onto voxel grid of output volume
def resampleSegments(voxel_data, positions, pixel_spacings):

    # TODO: Replace interpolation with pytorch one

    # R: segment positions off grid respective to output volume
    # W: segment positions on grid after resampling
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

        # No resampling if segment voxels are already aligned with output voxel grid
        doResample = (max_offset > c_resample_tolerance or voxel_count_dif != 0)

        result = None
        
        if doResample:

            if c_use_gpu:

                # Use numba implementation on gpu:
                scalings = (R_end[i, :] - R[i, :]) / dims[i, :]
                offsets = R[i, :] - W[i, :] 
                result = numba_interpolate.interpolate3d(W_size[i, :], voxel_data[i], scalings, offsets)

            else:
                # Use scipy CPU implementation:
                # Define positions of segment voxels (off of output volume grid)
                x_s = np.linspace(R[i, 0], R_end[i, 0], dims[i, 0])
                y_s = np.linspace(R[i, 1], R_end[i, 1], dims[i, 1])
                z_s = np.linspace(R[i, 2], R_end[i, 2], dims[i, 2])

                # Define positions of output volume voxel grid
                y_v = np.linspace(W[i, 0], W_end[i, 0], W_size[i, 0])
                x_v = np.linspace(W[i, 1], W_end[i, 1], W_size[i, 1])
                z_v = np.linspace(W[i, 2], W_end[i, 2], W_size[i, 2])

                xx_v, yy_v, zz_v = np.meshgrid(x_v, y_v, z_v)

                pts = np.zeros((xx_v.size, 3))
                pts[:, 1] = xx_v.flatten()
                pts[:, 0] = yy_v.flatten()
                pts[:, 2] = zz_v.flatten()

                # Resample segments onto output voxel grid
                rgi = scipy.interpolate.RegularGridInterpolator((x_s, y_s, z_s), voxel_data[i], bounds_error=False, fill_value=None)
                result = rgi(pts)

        else:
            # No resampling necessary
            result = voxel_data[i]

        result_data.append(result.reshape(W_size[i, :]))

    return (result_data, W, W_end, W_size)


def groupSlicesToSegments(slice_pixel_data, slice_series, slice_names, slice_positions, slice_pixel_spacings, slice_times):

    # Group by series into segments
    unique_series = np.unique(slice_series)

    #
    segment_voxel_data = []
    segment_series = []
    segment_names = []
    segment_positions = []
    segment_voxel_spacings = []
    segment_times = []

    # Each series forms one segment
    for s in unique_series:

        # Get slice indices for series s
        indices_s = [f for f, x in enumerate(slice_series) if x == s]

        # Get physical positions of slices
        slice_positions_s = [x for f, x in enumerate(slice_positions) if f in indices_s]

        position_max = np.amax(np.array(slice_positions_s).astype("float"), axis=0)
        segment_positions.append(position_max)

        # Combine slices to segment
        voxel_data_s = slicesToSegmentData(indices_s, slice_positions_s, slice_pixel_data)
        segment_voxel_data.append(voxel_data_s)

        # Get index of first slice
        slice_0 = indices_s[0]

        segment_series.append(slice_series[slice_0])
        segment_names.append(slice_names[slice_0])
        segment_times.append(slice_times[slice_0])

        # Get 3d voxel spacing
        voxel_spacing_2d = slice_pixel_spacings[slice_0]

        # Get third dimension by dividing segment extent by slice count
        z_min = np.amin(np.array(slice_positions_s)[:, 2].astype("float"))
        z_max = np.amax(np.array(slice_positions_s)[:, 2].astype("float"))
        z_spacing = (z_max - z_min) / (len(slice_positions_s) - 1)

        voxel_spacing = np.hstack((voxel_spacing_2d, z_spacing))
        segment_voxel_spacings.append(voxel_spacing)

    return (segment_voxel_data, segment_names, segment_positions, segment_voxel_spacings, segment_times)


def getDataFromDicom(ds):

    pixel_data = ds.pixel_array

    series = ds.get_item(["0020", "0011"]).value
    series = int(series)

    name = ds.get_item(["0008", "103e"]).value

    position = ds.get_item(["0020", "0032"]).value 
    position = np.array(position.decode().split("\\")).astype("float32")

    pixel_spacing = ds.get_item(["0028", "0030"]).value
    pixel_spacing = np.array(pixel_spacing.decode().split("\\")).astype("float32")

    start_time = ds.get_item(["0008", "0031"]).value

    return (pixel_data, series, name, position, pixel_spacing, start_time)


def slicesToSegmentData(slice_indices, slice_positions, slices):

    # Get size of output volume segment
    slice_count = len(slice_indices)
    slice_shape = slices[slice_indices[0]].shape

    # Get slice positions
    slices_z = np.zeros(slice_count)
    for z in range(slice_count):
        slices_z[z] = slice_positions[z][2]

    # Sort slices by position
    (slices_z, slice_indices) = zip(*sorted(zip(slices_z, slice_indices), reverse=True))

    # Write slices to volume segment
    dim = np.array((slice_shape[0], slice_shape[1], slice_count))
    segment = np.zeros(dim)

    for z in range(dim[2]):
        slice_z_index = slice_indices[z]
        segment[:, :, z] = slices[slice_z_index]

    return segment


def segmentsFromDicom(input_path_zip):

    # Get slice info
    pixel_data = []
    series = []
    names = []
    positions = []
    pixel_spacings = []
    times = []

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

    (segment_voxel_data, segment_names, segment_positions, segment_voxel_spacings, segment_times) = groupSlicesToSegments(pixel_data, series, names, positions, pixel_spacings, times)

    return (segment_voxel_data, segment_names, segment_positions, segment_voxel_spacings, segment_times)


##################################################
##################################################
# OLD:
##################################################
##################################################
    
def writeNrrd(volume, output_path, origin, version_tag):

    # See: http://teem.sourceforge.net/nrrd/format.html
    header = {'dimension': 3}
    header['type'] = c_datatype_nrrd
    header['sizes'] = volume.shape
    header['pipeline_version'] = version_tag

    # Spacing info compatible with 3D Slicer
    header['space dimension'] = 3
    header['space directions'] = np.array(c_out_pixel_spacing * np.eye(3,3))
    header['space origin'] = origin
    header['space units'] = "\"mm\" \"mm\" \"mm\""

    # Choose nrrd compression (both lossless, bzip2 at level 9 seems best)
    # Tested on one patient, generating both signal and fraction images:
    #   gzip, level 0 (100% size, 100% runtime), no compression
    #   gzip, level 1 ( 45% size, 200% runtime), fastest
    #   bzip2,level 9 ( 32% size, 650% runtime), smallest (can not be opened in 3d slicer?)
    #   bzip2,level 1 ( 33% size, 650% runtime)
    #   gzip, level 9 ( 40% size,5400% runtime), too slow
    # Runtime describes file output only (about 8s for level 0)
    #header = collections.OrderedDict()
    #header['encoding'] = 'bzip2' # alternative: 'gzip'
    header['encoding'] = 'gzip'
    compression_level = 1

    #
    nrrd.write(output_path + ".nrrd", volume, header, compression_level=compression_level)

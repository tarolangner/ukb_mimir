import os
import sys
import numpy as np
import shutil

## Write text files of subject ids and values for one regression target, subdivided by a cross-validation split.
# Only use subjects for whom both a valid target value and an existing input image are found.
def main(argv):

    imageset_path = "/home/taro/DL_imagesets/UKB_dual_projFF_uint8/"
    split_path = "../splits/mimir_72_10fold/"

    field = "22409-2.0"
    field_path = f"/media/taro/DATA/Taro/UKBiobank/extracted/fields/extracted_column_{field}.txt"
    output_path = "../targets/UKB_dual_projFF_uint8_AtlasThighMuscle_10fold/"

    formatTarget(imageset_path, split_path, output_path, field_path)

    # Write documenting properties to be listed in network evaluation
    description = "AtlasThighMuscle"
    unit = "L"
    with open(output_path + "documentation.txt", "w") as f:
        f.write("description,field,unit\n")
        f.write("{},{},{}\n".format(description, field, unit))


def formatTarget(imageset_path, split_path, output_path, field_path):

    print("Getting images and extracting labels...")
    image_ids = getImageIds(imageset_path + "/")

    (images, values) = extractLabels(image_ids, field_path)

    # Read training/validation split
    split_ids = readSplit(split_path)

    # Write output
    print("Generating output files...")
    writeDataset(output_path, images, values, split_ids, imageset_path, field_path)


def writeDataset(output_path, image_names, values, split_ids, imageset_path, field_path):

    if os.path.exists(output_path): 
        print(f"WARNING: Removing already existing directory at {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path)
    os.makedirs(output_path + "subsets/")

    values_all = []
    ids_all = []

    # Write ids and values to split subsets
    K = len(split_ids)
    for k in range(K):

        label_file = open(output_path + "subsets/subset_{}_labels.txt".format(k), "w")
        img_file = open(output_path + "subsets/subset_{}_ids.txt".format(k), "w")

        mask = np.in1d(image_names, split_ids[k])
        image_names_k = image_names[mask]
        values_k = values[mask]

        values_all.extend(values_k)
        ids_all.extend(image_names_k)

        for j in range(len(image_names_k)):
            img_file.write("{}\n".format(image_names_k[j]))
            label_file.write("{}\n".format(values_k[j]))
                    
        #
        label_file.close()
        img_file.close()

    N = len(np.unique(ids_all))

    print("  {} unique ids are part of dataset".format(N))


    if False:
        # Write image directory name
        images_path_file = open(output_path + "imageset_path.txt", "w")
        images_path_file.write(imageset_path)
        images_path_file.close()

        # Calculate metrics
        std = np.std(values, ddof=1)
        metrics_file = open(output_path + "metrics.txt", "w")
        metrics_file.write("stdev: {}\n".format(std))
        metrics_file.write("Range: [{}, {}]\n".format(np.amin(values), np.amax(values)))
        metrics_file.write("N: {}\n".format(N))
        metrics_file.write("id: {}\n".format(label_index))
        metrics_file.write("label: {}\n".format(label_name))
        metrics_file.close()


def readSplit(split_path):
    
    split_files = [f for f in os.listdir(split_path) if os.path.isfile(os.path.join(split_path, f))]
    split_files = [f for f in split_files if "images_set_" in f]

    K = len(split_files)

    split_ids = []
    ids_all = []

    for k in range(K):

        split_file_path = split_path + split_files[k]

        with open(split_file_path) as f:
            entries = f.read().splitlines()

        split_ids_k = np.array(entries).astype("int")
        split_ids.append(split_ids_k)

        ids_all.extend(split_ids_k)

    print("  {} ids are used in split".format(len(ids_all)))

    return split_ids


def getImageIds(imageset_path):

    files = [f for f in os.listdir(imageset_path) if os.path.isfile(os.path.join(imageset_path, f))]
    files = [f.replace(".npy", "") for f in files]

    ids = np.array(files).astype("int")

    return ids


def extractLabels(img_ids, field_path):

    # Read field
    with open(field_path) as f: entries = f.read().splitlines()
    entries.pop(0)

    # Format field values and remove invalid entries
    entries = [f.replace("\"","") for f in entries if not "nan" in f]

    # Parse field
    field_ids = np.array([f.split(",")[0] for f in entries]).astype("int")
    field_values = np.array([f.split(",")[1] for f in entries]).astype("float")

    #
    print("  Found {} valid labels and {} images".format(len(field_ids), len(img_ids)))

    # Get intersection
    mask = np.in1d(field_ids, img_ids)
    field_ids = field_ids[mask]
    field_values = field_values[mask]

    print("  For {} valid labels, an image was found".format(len(field_ids)))

    return (field_ids, field_values)


if __name__ == '__main__':
    main(sys.argv)

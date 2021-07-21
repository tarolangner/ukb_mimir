import os
import sys
import shutil

## Create an inference module consisting of:
# Network snapshot
# Metadata for documentation
# Standardization parameters for input scaling
# Calibration factors for scaling of predicted uncertainty
def main(argv):

    # Path to inference network (typically trained in train/test split)
    path_network = "/media/taro/DATA/Taro/Projects/mimir/ukb_mimir/cross_validation/networks/Mimir_m4_traintest/"
    path_snapshot = path_network + "subset_0/snapshots/snapshot_10000.pth.tar"

    # Path to calibration factors (typically trained in separate cross-validation)
    path_calibration = "/media/taro/DATA/Taro/Projects/mimir/ukb_mimir/cross_validation/networks/Mimir_m4_10fold/calibration_factors.txt"

    # Path to resulting inference module
    path_out = "module_test"

    ##
    # Avoid overwriting
    if os.path.exists(path_out): 
        print("ABORT: Output path already exists!")
        sys.exit()
    else:
        os.makedirs(path_out)

    # Copy trained network weights
    shutil.copyfile(path_snapshot, path_out + "/snapshot.pth.tar")

    # Copy standardization parameters
    path_stand = os.path.dirname(path_snapshot) + "/stand.txt"
    shutil.copyfile(path_stand, path_out + "/standardization_parameters.txt")

    # Write metadata
    writeMetadata(path_network, path_out)

    # Copy calibration parameters
    copyCalibrationFactors(path_calibration, path_out) 


# Get target name, field and unit
def parseTargetDocumentation(target_path):

    path_doc = target_path + "documentation.txt"
    with open(path_doc) as f: entries = f.readlines()

    name = entries[1].split(",")[0]
    field = entries[1].split(",")[1]
    unit = entries[1].split(",")[2].replace("\n", "")

    return (name, field, unit)


# Write "metadata.txt" with additional info on each regression target
def writeMetadata(path_network, path_out):

    path_target_list = path_network + "target_paths.txt"
    with open(path_target_list) as f: entries = f.readlines()
    target_paths = [f.replace("\n", "") + "/" for f in entries]

    ## Write metadata file
    with open(path_out + "/metadata.txt", "w") as f:

        # Note: The divisor can be changed manually,
        # for example to represent mL in L
        f.write("Name,Field,Unit,Divisor\n")

        #
        for t in range(len(target_paths)):
            (name_t, field_t, unit_t) = parseTargetDocumentation(target_paths[t])
            f.write(f"{name_t},{field_t},{unit_t},1\n")


# Copy target-wise factors that scale the uncertainty (predicted variances)
# thus that they are calibrated.
# Note: 
#   This framework fits the factors against validation data. 
#   The input path should therefore refer to a cross-validation
def copyCalibrationFactors(path_calibration, path_out):

    # Use factors from final snapshot
    snapshot_index = -1 

    #
    with open(path_calibration) as f: entries = f.readlines()
    factors = entries[snapshot_index].split(",")[1:]

    #
    with open(path_out + "/calibration_factors.txt", "w") as f:
        f.write("calibration_factor")
        for t in range(len(factors)):
            f.write(f"\n{factors[t]}")


if __name__ == '__main__':
    main(sys.argv)

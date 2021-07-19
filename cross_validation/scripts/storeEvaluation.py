import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import scipy

import plot_compare

from sklearn.metrics import r2_score

## For the given subset of the cross-validation:
# -store the sample-wise predictions of each network snapshot
# -store snapshot-wise evaluation metrics
# Note: values_means and values_vars have shape (N, T, S) as in (sample, target, snapshot)
def storePredictions(img_names, values_gt, values_means, values_vars, I, save_step, output_path, data_path, stands):

    # Revert label standardization
    T = len(stands)
    for t in range(T):

        # Ground truth and predicted means
        values_gt[:, t] = values_gt[:, t] * float(stands[t]["stdev"]) + float(stands[t]["mean"])
        values_means[:, t, :] = values_means[:, t, :] * float(stands[t]["stdev"]) + float(stands[t]["mean"])

        # Predicted variances
        values_vars[:, t, :] = values_vars[:, t, :] * np.square(float(stands[t]["stdev"]))

    # Get number of samples and snapshots
    #N = len(img_names)
    S = int(I // save_step)

    #
    iterations = np.zeros(S).astype("int")

    #
    r2 = np.zeros((S, T))
    icc = np.zeros((S, T))
    mae = np.zeros((S, T))
    mape = np.zeros((S, T))

    # Process each snapshot
    for s in range(S):

        # Get iteration at which snapshot was saved
        iterations[s] = int((s+1) * save_step)

        # Write sample-wise predictions of snapshot s for all targets t
        for t in range(T):
            metrics_path = output_path + "output_it_{}_t{}.txt".format(iterations[s], t)
            writePredictions(metrics_path, img_names, values_gt[:, t], values_means[:, t, s], values_vars[:, t, s])

        # Get evaluation metrics for snapshot s on all targets t
        for t in range(T):
            mask = np.invert(np.isnan(values_gt[:, t]))
            (r2[s, t], icc[s, t], mae[s, t], mape[s, t]) = getAgreementMetrics(values_gt[mask, t], values_means[mask, t, s])

    # Write validation curve and snapshot-wise evaluation metrics for all targets t
    for t in range(T):

        plotCurve(icc[:, t], "ICC", iterations, output_path + "validation_curve_t{}.png".format(t))

        #
        with open(output_path + f"snapshot_metrics_t{t}.txt", "w") as f:
            f.write("iteration,icc,r2,mae,mape\n")
            for s in range(S):
                f.write("{},{},{},{},{}\n".format(iterations[s], icc[s, t], r2[s, t], mae[s, t], mape[s, t]))
            

def parseTargetDocumentation(target_path):

    path_doc = target_path + "documentation.txt"
    with open(path_doc) as f: entries = f.readlines()

    name = entries[1].split(",")[0]
    field = entries[1].split(",")[1]
    unit = entries[1].split(",")[2].replace("\n", "")

    return (name, field, unit)


## Combine the predictions of all cross-validation subsets,
# write aggregated, sample-wise predictions
# calculate aggregated cross-validation metrics and visualize with plots
def aggregateValidation(output_path, target_paths, I, save_step, runtime):

    # Get number of targets and snapshots
    T = len(target_paths)
    S = int(I // save_step)

    #
    if not os.path.exists(output_path + "plot_agreement/"): 
        os.makedirs(output_path + "plot_agreement/")

    if not os.path.exists(output_path + "plot_uncertainty/"): 
        os.makedirs(output_path + "plot_uncertainty/")

    # Reserve validation metrics (snapshots, targets)
    r2 = np.zeros((S, T))
    icc = np.zeros((S, T))
    mae = np.zeros((S, T))
    mape = np.zeros((S, T))

    # Number of samples per target
    counts = np.zeros(T).astype("int")

    # Aggregate per target
    for t in range(T):

        (name_t, field_t, unit_t) = parseTargetDocumentation(target_paths[t])

        # Combine the prediction from each split and get number of total samples
        N = combinePredictions(output_path, I, save_step, t)
        counts[t] = N

        # Prepare ground truth and predictions
        values_gt = np.zeros(N)
        values_out_means = np.zeros((S, N))
        values_out_vars = np.zeros((S, N))

        #
        iterations = np.zeros(S).astype("int")

        # Read and evaluate aggregated predictions for each snapshot (saving step)
        for s in range(S):

            iterations[s] = int((s+1) * save_step) 

            (img_names, values_gt[:], values_out_means[s, :], values_out_vars[s, :]) = readFileMetrics(output_path + "/eval/output_it_{}_t{}.txt".format(iterations[s], t))
            (r2[s, t], icc[s, t], mae[s, t], mape[s, t]) = getAgreementMetrics(values_gt, values_out_means[s, :])

        # Plot validation curve
        plotCurve(icc[:, t], "ICC", iterations, output_path + "validation_curve_t{}_{}.png".format(t, name_t))

        # Generate plots
        plotAgreement(iterations, values_gt, values_out_means, output_path, t, name_t, unit_t)
        plotUncertaintyEval(iterations, values_gt, values_out_means, values_out_vars, output_path, t, name_t, unit_t)

        # Write aggregate metrics for given target per snapshot
        with open(output_path + f"snapshot_metrics_t{t}.txt", "w") as f:
            f.write("iteration,icc,r2,mae,mape\n")
            for s in range(S):
                f.write("{},{},{},{},{}\n".format(iterations[s], icc[s, t], r2[s, t], mae[s, t], mape[s, t]))

    # Write target-wise aggregate metrics for final network snapshots
    writeSummaryMetrics(output_path, target_paths, counts, icc, r2, mae, mape)


# Write target-wise evaluation metrics for final iteration
def writeSummaryMetrics(output_path, target_paths, counts, icc, r2, mae, mape):

    T = len(target_paths)

    with open(output_path + f"evaluation_summary.txt", "w") as f:
        f.write("name,field,N,unit,icc,r2,mae,mape\n")

        #
        for t in range(T):

            (name, field, unit) = parseTargetDocumentation(target_paths[t])

            f.write("{},{},{},{},".format(name, field, counts[t], unit))
            
            f.write("{},{},{},{}\n".format(r2[-1, t], icc[-1, t], mae[-1, t], mape[-1, t]))


# Get intraclass correlation coefficient
def getIcc(gt, out):

    N = len(gt)
    K = 2
    mat = np.transpose(np.vstack((gt, out)))

    ss_r = np.sum(np.square(np.mean(mat, axis = 1) - np.mean(mat))) * K
    ms_r = ss_r / (N-1) # / K

    ss_c = np.sum(np.square(np.mean(mat, axis = 0) - np.mean(mat))) # * N
    ms_c = ss_c / (K-1) # / N

    ss_e = 0
    for k in range(K):
        dif = mat[:, k] - np.mean(mat, axis=1)
        ss_e += np.sum(np.square(dif))

    ms_e = ss_e / ((N-1) * (K-1))

    icc_2_1 = (ms_r - ms_e) / (ms_r + (K - 1) * ms_e + K/N * (ms_c - ms_e))

    return icc_2_1


#
def getAgreementMetrics(gt, out):

    mask = np.invert(np.isnan(gt))
    gt = gt[mask]
    out = out[mask]

    if len(gt) > 1:

        mae = np.mean(np.abs(gt - out))    

        # Robust division without zeros
        zero_dif = np.abs(gt)
        epsilon = 0.0001
        mask = zero_dif > epsilon
        mape = np.mean(np.abs(np.abs(gt[mask] - out[mask]) / gt[mask]))

        r2 = r2_score(gt, out)
        icc = getIcc(gt, out)

    else:
        r2 = 0
        icc = 0
        mae = 0
        mape = 0

    return (r2, icc, mae, mape)


#
def plotAgreement(iterations, values_gt, values_out, network_path, t, name, unit):

    S = len(iterations)

    for s in range(S):

        path_out = network_path + "plot_agreement/plot_agreement_t{}_it_{}_{}.png".format(t, int(iterations[s]), name)

        #
        mask = np.invert(np.isnan(values_gt))
        plot_compare.plotScatter(values_gt[mask], values_out[s][mask], path_out, name, unit)


def plotCurve(values, metric_name, iterations, out_path):

    f, ax = plt.subplots(1)

    ax.plot(iterations, values, marker=".")

    plt.xlabel("iteration")

    ax.set_title("Validation")
    ax.grid()
    ax.set_ylabel(metric_name, color='C0')
    ax.tick_params('y', colors='C0')

    f.set_size_inches(5, 4, forward=True)
    f.tight_layout()

    f.savefig(out_path, dpi=100)

    plt.close()


def writePredictions(output_path, img_names, values_gt, values_means, values_vars):

    with open(output_path, "w") as f: 

        f.write("name,gt,out_mean,out_var\n")

        for i in range(len(values_gt)):
            f.write("{},{},{},{}\n".format(img_names[i], values_gt[i], values_means[i], values_vars[i]))


def combinePredictions(output_path, I, save_step, t):

    #
    dataset_names = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    dataset_names = [f for f in dataset_names if "subset_" in f]

    K = len(dataset_names)
    S = int(I // save_step)

    #
    for s in range(S):
        
        iteration = int((s+1) * save_step) 

        # Get outputs of split datasets
        img_names = []
        values_gt = []
        values_out_means = []
        values_out_vars = []

        for k in range(K):

            file_path = output_path + dataset_names[k] + "/eval/output_it_{}_t{}.txt".format(iteration, t)

            (img_names_k, values_gt_k, values_out_means_k, values_out_vars_k) = readFileMetrics(file_path)

            img_names.extend(img_names_k)

            values_gt.extend(values_gt_k)
            values_out_means.extend(values_out_means_k)
            values_out_vars.extend(values_out_vars_k)

        metrics_path = output_path + "eval/output_it_{}_t{}.txt".format(iteration, t)
        writePredictions(metrics_path, img_names, values_gt, values_out_means, values_out_vars)

    #
    N = len(values_gt)
    return N


def readFileMetrics(file_path):

    with open(file_path) as f:
        entries = f.read().splitlines()

    entries.pop(0)

    N = len(entries)

    #
    img_names = []
    values_gt = np.zeros(N)
    values_out_means = np.zeros(N)
    values_out_vars = np.zeros(N)

    #
    for i in range(N):
        parts = entries[i].split(",")

        img_names.append(parts[0])
        values_gt[i] = float(parts[1])
        values_out_means[i] = float(parts[2])
        values_out_vars[i] = float(parts[3])

    return (img_names, values_gt, values_out_means, values_out_vars)


def getCalibrationCurve(abs_difs, pred_vars):

    # 
    z_scores = np.abs(abs_difs / np.sqrt(pred_vars))

    values_p = np.zeros(10)
    fractions = np.zeros(10)

    N = len(abs_difs)

    #
    for i in range(10):
        
        interval = i + 1
        p = interval / float(10)

        values_p[i] = p 

        area = 0.5 + (p / 2)
        z = scipy.stats.norm.ppf(area)

        #print("P: {}, area: {}, z: {}".format(p, area, z))

        mask = z_scores <= z
        fractions[i] = np.sum(mask)

        #print("p: {}, Area: {}, z: {}".format(p, area, z))

    fractions /= N

    values_p = np.hstack((0, values_p))
    fractions = np.hstack((0, fractions))

    return (values_p, fractions)


def getAuce(values_p_pred, fractions_pred):

    auce = np.trapz(np.abs(values_p_pred - fractions_pred), dx = values_p_pred[1] - values_p_pred[0])

    return auce


def plotCalibrationCurves(iterations, values_gt, values_out_means, values_out_vars, output_path, t, s, ax):

    mask = np.invert(np.isnan(values_gt))

    (values_p, fractions) = getCalibrationCurve(np.abs(values_gt[mask] - values_out_means[s, mask]),  values_out_vars[s, mask])

    auce = getAuce(values_p, fractions)

    ax.plot([0, 1], [0, 1], linestyle="--", c="lightgray")
    ax.plot(values_p, fractions, marker=".", c="C0")
    ax.set_title("AUCE: {0:0.3f}".format(auce))


def getSparsificationCurve(values_gt, values_out_means, values_out_vars):

    idx = np.argsort(values_out_vars)[::-1]

    out_y = np.zeros(len(idx))
    for i in range(len(idx)):
        out_y[i] = np.mean(np.abs(values_gt[idx[i:]] - values_out_means[idx[i:]]))

    return out_y


def plotSparsificationCurves(iterations, values_gt, values_out_means, values_out_vars, output_path, t, s, ax):

    mask = np.invert(np.isnan(values_gt))

    points = getSparsificationCurve(values_gt[mask], values_out_means[s, mask], values_out_vars[s, mask])
    plt.plot(points)


def plotUncertaintyEval(iterations, values_gt, values_out_means, values_out_vars, output_path, t, name, unit):

    for s in range(len(iterations)):

        f, ax = plt.subplots(1, 2, figsize=(2*5.5, 5.5))

        plotCalibrationCurves(iterations, values_gt, values_out_means, values_out_vars, output_path, t, s, ax[0])
        plotSparsificationCurves(iterations, values_gt, values_out_means, values_out_vars, output_path, t, s, ax[1])

        plt.tight_layout()
        path_out = output_path + "plot_uncertainty/plot_uncertainty_t{}_it_{}_{}.png".format(t, int(iterations[s]), name)
        plt.savefig(path_out)
        plt.close()

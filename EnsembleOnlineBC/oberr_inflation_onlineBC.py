"""
Observation error inflation for online bias correction.
"""

import argparse
import shutil, os
import numpy as np
import h5py
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ch_to_idx(filename, ch_used):
    with h5py.File(filename, 'r') as f:
        channels = f['Channel'][:].tolist()
        ch_idx = [channels.index(ch) for ch in ch_used]
    return ch_idx


def read_obsout(filename, ch_idx, pred_list, n_ens, n_eig):
    n_pred = len(pred_list)
    n_chan = len(ch_idx)

    with h5py.File(filename, 'r') as f:
        fill_value = f['ObsError']['brightnessTemperature'].attrs["_FillValue"]
        obserrstd = f['ObsError']['brightnessTemperature'][:]
        n_loc, n_chan_all = obserrstd.shape

        hofx = np.zeros((n_loc, n_chan, n_ens, n_eig))
        predictors = np.zeros((n_loc, n_chan, n_pred))

        for pp, pred in enumerate(pred_list):
            predictors[:, :, pp] = f[pred + 'Predictor']['brightnessTemperature'][:, ch_idx]

        for ens in range(1, n_ens + 1):
            for eig in range(1, n_eig + 1):
                group_name = f"hofxm0_{eig}_{ens}"
                hofx[:, :, ens - 1, eig - 1] = f[group_name]['brightnessTemperature'][:, ch_idx]

    tmp = np.var(hofx, axis=(2, 3))
    # correction term (2026/05/13)
    var_hofx = tmp * (n_ens*n_eig -1)/(n_ens-1)
    mean_hofx = np.mean(hofx, axis=(2, 3))

    return obserrstd, predictors, var_hofx, mean_hofx, fill_value, n_loc, n_chan_all


def read_satbias(filename, ch_idx, pred_list):
    n_chan = len(ch_idx)
    n_pred = len(pred_list)

    coeff = np.zeros((n_chan, n_pred))
    coeff_errvar = np.zeros((n_chan, n_pred))

    with h5py.File(filename, 'r') as f:
        for pp, pred in enumerate(pred_list):
            coeff[:, pp] = f['BiasCoefficients'][pred][:]
            coeff_errvar[:, pp] = f['BiasCoefficientErrors'][pred][:]

    return coeff, coeff_errvar


def compute_inflation(obserrstd, predictors, coeff_errvar, ch_idx, fill_value):
    n_loc, n_chan = predictors.shape[0], predictors.shape[1]
    inflation = np.zeros((n_loc, n_chan))

    for ll in range(n_loc):
        for cc in range(n_chan):
            p_vec = predictors[ll, cc, :].reshape(1, -1)
            B_mat = np.diag(coeff_errvar[cc, :])
            inflation[ll, cc] = (p_vec @ B_mat @ p_vec.T).item()

    obserrstd_inflated = np.sqrt(obserrstd[:, ch_idx] ** 2 + inflation)

    mask = obserrstd[:, ch_idx] == fill_value
    obserrstd_inflated[mask] = fill_value

    return obserrstd_inflated


def write_output(filename, filename_out, obserrstd, obserrstd_inflated, var_hofx,
                 ch_idx, fill_value, n_loc, n_chan_all):

    with h5py.File(filename_out, 'r+') as f:
        f['ObsError']['brightnessTemperature'][:, ch_idx] = obserrstd_inflated

        grp = f.require_group("ObsError_orig")
        if "brightnessTemperature" in grp:
            del grp["brightnessTemperature"]
        dset = grp.create_dataset("brightnessTemperature", data=obserrstd)
        dset.dims[0].attach_scale(f["Location"])
        dset.dims[1].attach_scale(f["Channel"])
        dset.attrs["_FillValue"] = fill_value
        dset.attrs["units"] = "K"
        dset.attrs["description"] = (
            "Original observation error standard deviation (before online BC inflation)"
        )

        grp = f.require_group("prior_HofX_variance")
        if "brightnessTemperature" in grp:
            del grp["brightnessTemperature"]
        tmp = np.full((n_loc, n_chan_all), fill_value)
        tmp[:, ch_idx] = var_hofx
        dset = grp.create_dataset("brightnessTemperature", data=tmp)
        dset.dims[0].attach_scale(f["Location"])
        dset.dims[1].attach_scale(f["Channel"])
        dset.attrs["_FillValue"] = fill_value
        dset.attrs["units"] = "K^2"
        dset.attrs["description"] = "prior ensemble variance in obs space (HBH^T)"


def main(config_path):

    # YAML configurations:
    # n_ens  = ensemble size
    # n_eig  = # of eigenvalues in LGETKF
    # ch_used = the used channels (currently tested only for AHI)
    cfg = load_config(config_path)

    filename = cfg['files']['obsout_infile']          # obsout*.h5 (input)
    filename_bias = cfg['files']['satbias_infile']    # satbias*.h5 (input)
    filename_out = cfg['files']['obsout_outfile']     # updated obsout*.h5 (output)


    if not os.path.exists(filename_out):
        shutil.copy2(filename, filename_out)

    if filename == filename_out: # make a copy of the original one
        base, ext = os.path.splitext(filename)
        filename_orig = base + "_orig" + ext
        
        # to ensure not inflating twice, if _orig.h5 exists (meaning it has already been implemented before)
        # use the _orig.h5 as the filename
        if os.path.exists(filename_orig):
            filename = filename_orig
        else:
            shutil.copy2(filename, filename_orig)


    pred_list = cfg['predictors']
    n_ens = cfg['ensemble']['n_ens']
    n_eig = cfg['ensemble']['n_eig']
    ch_used = cfg['channels']['ch_used']

    # convert used channels to indices:
    ch_idx = ch_to_idx(filename, ch_used)

    # read input obsout*.h5 file: 
    # n_loc      = # of locations
    # n_pred     = # of predictors
    # n_chan     = # of used channels ( = len(ch_used) )
    # n_chan_all = # of all channels in obsout*.h5
    # obserrstd  = stdev of obs error (n_loc, n_chan_all)
    # predictors = predictors (n_loc, n_chan, n_pred)
    # var_hofx   = variance of prior ensemble H(x), or HBHT (n_loc, n_chan)
    # mean_hofx  = mean of prior ensemble H(x) (n_loc, n_chan)
    obserrstd, predictors, var_hofx, mean_hofx, fill_value, n_loc, n_chan_all = \
        read_obsout(filename, ch_idx, pred_list, n_ens, n_eig)

    # read input satbias*.h5 file:
    # coeff        = bias coefficients beta (n_chan, n_pred) 
    # coeff_errvar = bias coefficients error variance (n_chan, n_pred)
    coeff, coeff_errvar = read_satbias(filename_bias, ch_idx, pred_list)

    # compute the effective obs error stdev (i.e., inflated errors due to online BC)
    # obserrstd_inflated (n_loc, n_chan)
    obserrstd_inflated = compute_inflation(obserrstd, predictors, coeff_errvar, ch_idx, fill_value)

    # write a new obsout*.h5 file (overwriting ObsError, adding ObsError_orig, prior_HofX_variance)
    write_output(filename, filename_out, obserrstd, obserrstd_inflated, var_hofx,
                 ch_idx, fill_value, n_loc, n_chan_all)

    print("Done. Output written to:", filename_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='oberr_inflation_onlineBC.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)

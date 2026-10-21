#!/usr/bin/env python3
'''
Joint VarBC estimator: all channels solved simultaneously with full HBH^T.

Key differences from mpas_jedi_online_satbias_estimator.py:
  1. All channels solved jointly (not independently per channel)
  2. Full HBH^T (N×N) used instead of diagonal — requires precomputed file
     from compute_ensemble_HBHt.py
  3. R̃ = diag(σ²_R) + HBH^T  (full N×N matrix)

Math:
    β      ∈ ℝ^{p·Nch}          stacked coefficients  [β_ch1, β_ch2, ...]
    P      ∈ ℝ^{N × p·Nch}     block-diagonal predictor matrix
    B      ∈ ℝ^{p·Nch × p·Nch} block-diagonal prior covariance
    R̃      ∈ ℝ^{N × N}         diag(σ²) + HBH^T   (used for β only)
    R      ∈ ℝ^{N × N}         diag(σ²)           (used for analysis variance)
    d      ∈ ℝ^N                stacked innovations (uncorrected OMB)

    A      = pinv(B⁻¹ + Pᵀ R̃⁻¹ P)          (used for β)
    β_new  = A (B⁻¹ β₀ + Pᵀ R̃⁻¹ d)
    A_var  = pinv(B⁻¹ + Pᵀ R⁻¹ P)              saved analysis variance (original R)

where N = Σ n_ch is the total number of QC-passing observations across channels.
Observations are stacked in the order given by channels_used.
'''

import numpy as np
np.set_printoptions(precision=4)
import h5py
import os, argparse, json
import yaml
from scipy.linalg import block_diag

# ── Globals set from YAML ────────────────────────────────────────────────────
qcGroupName       = None
obsGroupName      = None
hofxGroupName     = None
biasGroupName     = None
errGroupName      = None
#priorvarGroupName = None
user_defined_predictorNames   = None
channels_used                 = None
inflation_factor              = None
dataSelCri                    = None


class DataSelectionCriterion:
    def __init__(self, method, cloud_diff_threshold=0):
        if method not in ['cloud_consistent', 'default']:
            raise ValueError(f"Invalid method: {method}")
        self.method = method
        self.cloud_diff_threshold = cloud_diff_threshold


# ── ObsoutReader ─────────────────────────────────────────────────────────────

class ObsoutReader:
    '''
    Reads obsout HDF5 files and applies QC filtering.
    db_dict: {(groupname, channel): list of values}
    '''

    def __init__(self, fname):
        self.db_dict = {}
        with h5py.File(fname, 'r') as f:
            keys = list(f.keys())
            self.predictorNames = [k.replace('Predictor', '') for k in keys if k.endswith('Predictor')]
            self.channels = list(f['Channel'][:])
        if user_defined_predictorNames is not None:
            self.predictorNames = user_defined_predictorNames
            print('Using user-defined predictors')
        print('Predictors:', self.predictorNames)

    def _read_group(self, hdf5, groupname):
        group   = hdf5[groupname]
        varname = group[list(group.keys())[0]].name
        return hdf5[varname][:, :]   # (n_loc, n_ch_total)

    def _compute_all_qc_indices(self, hdf5):
        '''QC boolean mask per channel. Returns list of bool arrays (one per channel).'''
        qc2d = self._read_group(hdf5, qcGroupName)
        if dataSelCri.method == 'cloud_consistent':
            clrBT   = self._read_group(hdf5, 'clearBkgBrightnessTemperaturePredictor')
            obsBT   = self._read_group(hdf5, 'ObsValue')
            modelBT = self._read_group(hdf5, hofxGroupName) - self._read_group(hdf5, biasGroupName)
            cld_obs   = np.maximum(clrBT - obsBT,   0)
            cld_model = np.maximum(clrBT - modelBT, 0)
        indices = []
        for ch in self.channels:
            ich = self.channels.index(ch)
            if dataSelCri.method == 'default':
                index = qc2d[:, ich] == 0
            elif dataSelCri.method == 'cloud_consistent':
                index = (
                    (qc2d[:, ich] == 0)
                    & (np.abs(cld_obs[:, ich] - cld_model[:, ich]) < dataSelCri.cloud_diff_threshold)
                    & (clrBT[:, ich] >= np.maximum(obsBT[:, ich], modelBT[:, ich]))
                )
            indices.append(index)
        return indices

    def _add_var_with_index(self, hdf5, groupname, indices):
        var2d = self._read_group(hdf5, groupname)
        for ch in self.channels:
            ich      = self.channels.index(ch)
            index    = indices[ich]
            var_qced = var2d[index, ich] if index.any() else np.array([])
            if (groupname, ch) not in self.db_dict:
                self.db_dict[(groupname, ch)] = list(var_qced)
            else:
                self.db_dict[(groupname, ch)].extend(list(var_qced))

    def add_file(self, fname):
        print('read', fname)
        with h5py.File(fname, 'r') as f:
            indices = self._compute_all_qc_indices(f)
            for key in self.predictorNames:
                self._add_var_with_index(f, key + 'Predictor', indices)
            #for key in [obsGroupName, hofxGroupName, biasGroupName,
            #            errGroupName, priorvarGroupName]:
            for key in [obsGroupName, hofxGroupName, biasGroupName,
                        errGroupName]:
                self._add_var_with_index(f, key, indices)

    def get_channels(self):
        return self.channels

    def get_predNames(self):
        return self.predictorNames

    # ── Per-channel accessors (kept for compatibility / diagnostics) ─────────

    def get_Predictor(self, channel):
        '''Returns (p, n_ch) array.'''
        res = [self.db_dict[(pn + 'Predictor', channel)] for pn in self.predictorNames]
        return np.array(res, dtype=np.float64)

    def get_Yd(self, channel):
        obs  = np.array(self.db_dict[(obsGroupName,  channel)], dtype=np.float64)
        hofx = np.array(self.db_dict[(hofxGroupName, channel)], dtype=np.float64)
        bias = np.array(self.db_dict[(biasGroupName, channel)], dtype=np.float64)
        return obs - hofx + bias   # uncorrected OMB

    def get_err(self, channel):
        return np.array(self.db_dict[(errGroupName, channel)], dtype=np.float64)

    # ── Joint assembly ───────────────────────────────────────────────────────

    def build_joint_data(self, satbias_io):
        '''
        Assemble stacked arrays for all channels_used.

        Returns
        -------
        d      : (N,)           stacked innovations
        P      : (N, p*Nch)     block-diagonal predictor matrix
        sigma2 : (N,)           obs error variances (diagonal of R)
        beta0  : (p*Nch,)       stacked prior coefficients
        B      : (p*Nch, p*Nch) block-diagonal prior covariance
        n_obs  : list[int]      n_ch for each channel (in channels_used order)
        '''
        d_list      = []
        P_list      = []
        sigma2_list = []
        beta0_list  = []
        B_list      = []
        n_obs       = []

        for ch in channels_used:
            d_ch      = self.get_Yd(ch)                                # (n_ch,)
            P_ch      = self.get_Predictor(ch).T                       # (n_ch, p)
            err_ch    = self.get_err(ch)                               # (n_ch,)
            beta0_ch  = satbias_io.getBeta(ch)                         # (p,)
            B_ch      = inflation_factor * satbias_io.getCov(ch)       # (p, p)

            d_list.append(d_ch)
            P_list.append(P_ch)
            sigma2_list.append(err_ch ** 2)
            beta0_list.append(beta0_ch)
            B_list.append(B_ch)
            n_obs.append(len(d_ch))

        d      = np.concatenate(d_list)        # (N,)
        P      = block_diag(*P_list)           # (N, p*Nch)
        sigma2 = np.concatenate(sigma2_list)   # (N,)
        beta0  = np.concatenate(beta0_list)    # (p*Nch,)
        B      = block_diag(*B_list)           # (p*Nch, p*Nch)

        return d, P, sigma2, beta0, B, n_obs


# ── SatbiasIO ────────────────────────────────────────────────────────────────

class SatbiasIO:
    def __init__(self, predictorNames, channels_used):
        nch = len(channels_used)
        self.predictorNames = predictorNames
        self.channels_used  = channels_used
        self.num_obs_used   = np.zeros(nch)
        self.beta_dict = {k: np.zeros(nch,  dtype=np.float64) for k in predictorNames}
        self.err_dict  = {k: 1e20 * np.ones(nch, dtype=np.float64) for k in predictorNames}


    def readin_satbias_beta(self, fname, groupname='BiasCoefficients'):
        if not os.path.exists(fname):
            print('Warning: satbias_input not found, starting from zero beta')
            return
        with h5py.File(fname, 'r') as hdf5:
            group = hdf5[groupname]
            for key in group.keys():
                var2d = hdf5[group[key].name]
                if groupname == 'BiasCoefficients':
                    self.beta_dict[key] = var2d[0, :]
                elif groupname == 'BiasCoefficientErrors':
                    self.err_dict[key]  = var2d[0, :]

    def readin_satbias_err(self, fname):
        self.readin_satbias_beta(fname, groupname='BiasCoefficientErrors')

    def getBeta(self, channel):
        ich = self.channels_used.index(channel)
        return np.array([self.beta_dict[k][ich] for k in self.predictorNames], dtype=np.float64)

    def getCov(self, channel):
        ich = self.channels_used.index(channel)
        return np.diag([self.err_dict[k][ich] for k in self.predictorNames])

    def setBeta(self, key, channel, value):
        self.beta_dict[key][self.channels_used.index(channel)] = value

    def setCov(self, key, channel, value):
        self.err_dict[key][self.channels_used.index(channel)] = value

    def setNumObsUsed(self, channel, value):
        self.num_obs_used[self.channels_used.index(channel)] = value

    def _attach_labels(self, f, varname, dim1, dim2):
        h = f[varname]
        h.dims[0].attach_scale(dim1)
        h.dims[1].attach_scale(dim2)

    def save(self, outname):
        shape = (1, len(self.channels_used))
        with h5py.File(outname, 'w') as f:
            f.attrs['_ioda_layout']         = 'ObsGroup'
            f.attrs['_ioda_layout_version'] = 0
            var_ch  = f.create_dataset('Channel', data=np.array(self.channels_used), dtype=np.int32)
            var_rec = f.create_dataset('Record',  data=np.arange(1),                  dtype=np.int32)
            var_n   = f.create_dataset('numberObservationsUsed',
                                       data=self.num_obs_used.reshape(shape), dtype=np.int32)
            self._attach_labels(f, 'numberObservationsUsed', var_rec, var_ch)
            g_beta = f.create_group('BiasCoefficients')
            g_err  = f.create_group('BiasCoefficientErrors')
            for key in self.beta_dict:
                g_beta.create_dataset(key, data=self.beta_dict[key].reshape(shape), dtype=np.float32)
                g_err.create_dataset( key, data=self.err_dict[key].reshape(shape),  dtype=np.float32)
                self._attach_labels(f, f'BiasCoefficients/{key}',     var_rec, var_ch)
                self._attach_labels(f, f'BiasCoefficientErrors/{key}', var_rec, var_ch)
        print('Saved satbias →', outname)


# ── JointBiasCorrection ──────────────────────────────────────────────────────

class JointBiasCorrection:
    '''Solves for bias coefficients across all channels simultaneously.'''

    def __init__(self, obsout_reader, satbias_io, HBHt, meta):
        self.reader    = obsout_reader
        self.satbias   = satbias_io
        self.HBHt      = HBHt
        self.meta      = meta
        self.predNames = obsout_reader.get_predNames()
        p   = len(self.predNames)
        Nch = len(channels_used)
        self.p   = p
        self.Nch = Nch

    def _verify_hbht_meta(self, n_obs):
        '''Check that HBH^T metadata matches current data.'''
        expected = self.meta['n_obs_per_channel']
        if expected != n_obs:
            raise ValueError(
                f'HBH^T metadata mismatch.\n'
                f'  expected n_obs_per_channel = {expected}\n'
                f'  got                        = {n_obs}'
            )
        if self.meta['channels_used'] != channels_used:
            raise ValueError(
                f'HBH^T channels mismatch: {self.meta["channels_used"]} vs {channels_used}'
            )

    def solve(self):
        '''
        Joint solve across all channels.

        Updates satbias_io in-place.
        Returns beta_new (p*Nch,) and A (p*Nch × p*Nch).
        '''
        d, P, sigma2, beta0, B, n_obs = self.reader.build_joint_data(self.satbias)
        self._verify_hbht_meta(n_obs)

        # ── Build R̃ = diag(σ²) + HBH^T  ───────────────────────────────────
        R_tilde = np.diag(sigma2) + self.HBHt          # (N, N)

        # ── Solve linear system instead of explicit inversion ───────────────
        # Pt_Rtinv = P^T R̃⁻¹  →  solve R̃ X = P, then X^T = P^T R̃⁻¹
        RtinvP    = np.linalg.solve(R_tilde, P)         # (N, p*Nch)
        Rtinv_d   = np.linalg.solve(R_tilde, d)         # (N,)

        B_inv     = np.linalg.pinv(B)                   # (p*Nch, p*Nch)
        hessian   = B_inv + P.T @ RtinvP                # (p*Nch, p*Nch)
        A         = np.linalg.pinv(hessian)             # used for beta only (R̃-based)

        rhs      = B_inv @ beta0 + P.T @ Rtinv_d        # (p*Nch,)
        beta_new = A @ rhs                               # (p*Nch,)

        # ── Analysis variance matrix ─────────────────────────────────────────
        # Hessian uses the original R = diag(σ²) (no HBH^T); beta above uses R̃
        A_var = np.linalg.pinv(B_inv + P.T @ (P / sigma2[:, None]))   # (p*Nch, p*Nch)

        # ── Store results back into satbias_io ──────────────────────────────
        for i_ch, ch in enumerate(channels_used):
            sl = slice(i_ch * self.p, (i_ch + 1) * self.p)
            beta_ch = beta_new[sl]
            A_var_ch = A_var[sl, sl]
            for i_p, key in enumerate(self.predNames):
                self.satbias.setBeta(key, ch, beta_ch[i_p])
                self.satbias.setCov( key, ch, A_var_ch[i_p, i_p])
            self.satbias.setNumObsUsed(ch, n_obs[i_ch])

        self._print_diagnostics(d, P, sigma2, beta0, beta_new, A, n_obs)
        return beta_new, A

    def _print_diagnostics(self, d, P, sigma2, beta0, beta_new, A, n_obs):
        print(f'\n{"ch":>4}  {"n_obs":>6}  {"MEAN(OMA_bc)":>12}  {"RMS(OMA_bc)":>11}  '
              f'{"mean_bias":>10}  beta  (each predictor)')
        offset = 0
        for i_ch, ch in enumerate(channels_used):
            n   = n_obs[i_ch]
            sl  = slice(i_ch * self.p, (i_ch + 1) * self.p)
            beta_ch = beta_new[sl]
            d_ch    = d[offset:offset + n]
            P_ch    = P[offset:offset + n, sl]
            obsbias = P_ch @ beta_ch
            resid   = d_ch - obsbias
            if n > 0:
                print(f'{ch:>4}  {n:>6}  {np.mean(resid):>12.4f}  '
                      f'{np.sqrt(np.mean(resid**2)):>11.4f}  '
                      f'{np.mean(obsbias):>10.4f}  '
                      + '  '.join(f'{b:.4f}' for b in beta_ch))
            else:
                print(f'{ch:>4}  {n:>6}  (no valid obs)')
            offset += n

    def save(self, outname):
        self.satbias.save(outname)


# ── CSV output ───────────────────────────────────────────────────────────────

def save_csv(output_fname, satbias_io, n_obs_dict):
    import pandas as pd
    rows = []
    for ch in channels_used:
        row = {'channel': ch, 'obs_num': n_obs_dict[ch]}
        for key in satbias_io.beta_dict:
            ich = satbias_io.channels_used.index(ch)
            row[f'beta_{key}']     = float(satbias_io.beta_dict[key][ich])
            row[f'beta_var_{key}'] = float(satbias_io.err_dict[key][ich])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_fname, index=False)
    print('Saved CSV →', output_fname)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint VarBC estimator (full HBH^T)')
    parser.add_argument('--config', default='mpas_jedi_online_satbias_estimator_joint.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Set globals ──────────────────────────────────────────────────────────
    qcGroupName       = cfg['group_names']['qc']
    obsGroupName      = cfg['group_names']['obs']
    hofxGroupName     = cfg['group_names']['hofx']
    biasGroupName     = cfg['group_names']['bias']
    errGroupName      = cfg['group_names']['obs_error']
    #priorvarGroupName = cfg['group_names']['prior_var']
    user_defined_predictorNames  = cfg.get('predictors', None)
    channels_used                = cfg['channels_used']
    inflation_factor             = cfg['inflation_factor']
    dataSelCri = DataSelectionCriterion(
        cfg['data_selection']['method'],
        cfg['data_selection']['cloud_diff_threshold'],
    )

    obsout_fnames  = cfg['files']['obsout']
    hbht_fname     = cfg['files']['hbht_input']
    satbias_input  = cfg['files']['satbias_input']
    satbias_output = cfg['files']['satbias_output']
    csv_output     = cfg['files'].get('csv_output', '')

    # ── Read obsout ──────────────────────────────────────────────────────────
    obsout = ObsoutReader(obsout_fnames[0])
    for fname in obsout_fnames:
        obsout.add_file(fname)

    # ── Load HBH^T ───────────────────────────────────────────────────────────
    meta_fname = hbht_fname.replace('.npy', '_meta.json')
    with open(meta_fname) as f:
        hbht_meta = json.load(f)
    HBHt = np.load(hbht_fname)
    print(f'Loaded HBH^T {HBHt.shape} from {hbht_fname}')

    # ── Read satbias prior ───────────────────────────────────────────────────
    predNames = obsout.get_predNames()
    satbias   = SatbiasIO(predNames, channels_used)
    satbias.readin_satbias_beta(satbias_input)
    satbias.readin_satbias_err(satbias_input)

    # ── Joint solve ──────────────────────────────────────────────────────────
    bc = JointBiasCorrection(obsout, satbias, HBHt, hbht_meta)
    beta_new, A = bc.solve()
    bc.save(satbias_output)

    # ── CSV ──────────────────────────────────────────────────────────────────
    if csv_output:
        import pandas as pd
        n_obs_dict = {ch: hbht_meta['n_obs_per_channel'][i]
                      for i, ch in enumerate(channels_used)}
        save_csv(csv_output, satbias, n_obs_dict)

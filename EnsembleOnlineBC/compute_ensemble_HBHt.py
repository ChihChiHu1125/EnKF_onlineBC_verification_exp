#!/usr/bin/env python3
'''
Estimate the full HBH^T matrix from ensemble member HofX stored in a single obsout file,
with optional Gaspari-Cohn distance-based localization.

Group naming convention in the obsout file:
  hofx0_<n>            original ensemble members  (n = 1 .. n_ens)
  hofxm0_<neig>_<n>   modulated members           (neig = 1 .. n_eig,  n = 1 .. n_ens)

The script auto-detects all matching groups unless the member type is restricted
via the YAML config (member_type: original | modulated | both).

Localization (optional):
  If localization.horizontal_scale is set in the YAML config, HBH^T is multiplied
  element-wise by a Gaspari-Cohn weight matrix computed from great-circle distances
  between observations.  horizontal_scale is the GC half-width c (km); the function
  reaches zero at distance = 2c.

Output:
    <hbht_output>.npy       float64 array, shape (N, N)
    <hbht_output>_meta.json channel list, n_obs per channel, K, member groups used
'''

import numpy as np
import h5py
import re, os, argparse, json
import yaml

# ── globals set from YAML ────────────────────────────────────────────────────
qcGroupName       = None
biasGroupName     = None   # needed only for cloud_consistent QC
hofxMeanGroupName = None   # needed only for cloud_consistent QC
channels_used     = None
dataSelCri        = None


class DataSelectionCriterion:
    def __init__(self, method, cloud_diff_threshold=0):
        if method not in ['cloud_consistent', 'default']:
            raise ValueError(f"Invalid method: {method}")
        self.method = method
        self.cloud_diff_threshold = cloud_diff_threshold


def gaspari_cohn(r):
    '''
    Gaspari-Cohn (1999) compactly-supported correlation function.
    r = distance / horizontal_scale  (normalized distance).
    Returns values in [0, 1]; reaches 0 at r = 2.
    Accepts NumPy arrays of any shape.
    '''
    gc   = np.zeros_like(r, dtype=np.float64)
    m1   = r <= 1.0
    m2   = (r > 1.0) & (r <= 2.0)
    z1   = r[m1]
    z2   = r[m2]
    gc[m1] = (  - (1/4) * z1**5 + (1/2) * z1**4
                + (5/8) * z1**3 - (5/3) * z1**2 + 1.0)
    gc[m2] = (    (1/12) * z2**5 - (1/2) * z2**4
                + (5/8) * z2**3 + (5/3) * z2**2
                - 5.0 * z2      + 4.0   - (2/3) / z2)
    return gc


def pairwise_haversine_km(lats, lons):
    '''
    Compute (N, N) great-circle distance matrix in km.
    lats, lons : 1-D arrays of length N (degrees).
    '''
    lat = np.radians(lats)
    lon = np.radians(lons)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_localization_matrix(lats, lons, horizontal_scale):
    '''
    Build N×N Gaspari-Cohn localization matrix from observation lat/lon.
    horizontal_scale : GC half-width c in km (GC → 0 at distance = 2c).
    '''
    dist_km = pairwise_haversine_km(lats, lons)
    r       = dist_km / horizontal_scale
    return gaspari_cohn(r)


def _read_group(hdf5, groupname):
    group   = hdf5[groupname]
    varname = group[list(group.keys())[0]].name
    return hdf5[varname][:, :]    # (n_loc, n_ch_total)


def detect_member_groups(hdf5, member_type):
    '''
    Scan all top-level keys and return sorted list of HofX group names
    and the number of original ensemble members n_ens.

    member_type : 'original'  → hofx0_<n>
                  'modulated'  → hofxm0_<neig>_<n>
                  'both'       → both of the above

    Returns (groups, n_ens) where n_ens is the ensemble size
    '''
    pat_orig = re.compile(r'^hofx0_(\d+)$')
    pat_mod  = re.compile(r'^hofxm0_(\d+)_(\d+)$')

    orig_groups = []
    mod_groups  = []
    for key in hdf5.keys():
        if pat_orig.match(key):
            orig_groups.append(key)
        elif pat_mod.match(key):
            mod_groups.append(key)

    # Sort numerically so the mean is reproducible
    orig_groups.sort(key=lambda k: int(pat_orig.match(k).group(1)))
    mod_groups.sort(key=lambda k: (int(pat_mod.match(k).group(1)),
                                   int(pat_mod.match(k).group(2))))

    # n_ens = number of original ensemble members (denominator is always n_ens - 1)
    n_ens_orig = len(orig_groups) if orig_groups else max(
        int(pat_mod.match(k).group(2)) for k in mod_groups
    )

    if member_type == 'original':
        groups = orig_groups
    elif member_type == 'modulated':
        groups = mod_groups
    else:   # 'both'
        groups = orig_groups + mod_groups

    return groups, n_ens_orig


def compute_qc_indices(hdf5, all_channels):
    '''
    Compute per-channel QC indices.
    Returns {ch: integer index array} of locations that pass QC.
    Using integer arrays makes n_obs counting unambiguous: len(qc_indices[ch]).
    '''
    qc2d = _read_group(hdf5, qcGroupName)

    if dataSelCri.method == 'cloud_consistent':
        clrBT   = _read_group(hdf5, 'clearBkgBrightnessTemperaturePredictor')
        obsBT   = _read_group(hdf5, 'ObsValue')
        modelBT = _read_group(hdf5, hofxMeanGroupName) - _read_group(hdf5, biasGroupName)
        cld_obs   = np.maximum(clrBT - obsBT,   0)
        cld_model = np.maximum(clrBT - modelBT, 0)

    qc_indices = {}
    for ch in channels_used:
        ich = all_channels.index(ch)
        if dataSelCri.method == 'default':
            mask = qc2d[:, ich] == 0
        elif dataSelCri.method == 'cloud_consistent':
            mask = (
                (qc2d[:, ich] == 0)
                & (np.abs(cld_obs[:, ich] - cld_model[:, ich]) < dataSelCri.cloud_diff_threshold)
                & (clrBT[:, ich] >= np.maximum(obsBT[:, ich], modelBT[:, ich]))
            )
        qc_indices[ch] = np.where(mask)[0]   # integer index array
    return qc_indices


def extract_latlon_vectors(hdf5, meta_group, qc_indices):
    '''
    Read QC-filtered latitude and longitude for the stacked observation vector.
    Returns (lats, lons) each of length N, ordered the same way as extract_hofx_vector.
    '''
    lats_all = hdf5[f'{meta_group}/latitude'][:]
    lons_all = hdf5[f'{meta_group}/longitude'][:]
    lats = np.concatenate([lats_all[qc_indices[ch]] for ch in channels_used])
    lons = np.concatenate([lons_all[qc_indices[ch]] for ch in channels_used])
    return lats.astype(np.float64), lons.astype(np.float64)


def extract_hofx_vector(hdf5, groupname, all_channels, qc_indices):
    '''
    Read QC-filtered HofX for channels_used from one group.
    Returns 1-D array of length N (obs stacked in channels_used order).
    '''
    hofx2d = _read_group(hdf5, groupname)
    parts  = []
    for ch in channels_used:
        ich = all_channels.index(ch)
        parts.append(hofx2d[qc_indices[ch], ich])
    return np.concatenate(parts).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description='Estimate HBH^T from ensemble HofX in a single obsout file')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    global qcGroupName, biasGroupName, hofxMeanGroupName, channels_used, dataSelCri
    qcGroupName       = cfg['group_names']['qc']
    biasGroupName     = cfg['group_names'].get('bias',      'ObsBias0')
    hofxMeanGroupName = cfg['group_names'].get('hofx_mean', 'hofx_y_mean_xb0')
    channels_used = cfg['channels_used']
    member_type   = cfg.get('member_type', 'modulated')   # 'original' | 'modulated' | 'both'
    diagonal_only = cfg.get('diagonal_only', False)
    dataSelCri    = DataSelectionCriterion(
        cfg['data_selection']['method'],
        cfg['data_selection']['cloud_diff_threshold'],
    )
    loc_cfg           = cfg.get('localization', {})
    horizontal_scale  = loc_cfg.get('horizontal_scale', None)   # km; None → no localization

    obsout_fname = cfg['files']['obsout']
    output_fname = cfg['files']['hbht_output']

    with h5py.File(obsout_fname, 'r') as f:

        # ── Step 1: Detect member groups ─────────────────────────────────────
        member_groups, n_ens = detect_member_groups(f, member_type)
        K = len(member_groups)
        if K == 0:
            raise RuntimeError(f"No HofX groups found for member_type='{member_type}'")

        all_channels = list(f['Channel'][:])

        print(f"Member type   : {member_type}")
        print(f"K members     : {K}  (e.g. {member_groups[0]} … {member_groups[-1]})")
        print(f"n_ens         : {n_ens}  →  denominator = n_ens - 1 = {n_ens - 1}")
        print(f"Channels      : {channels_used}")

        # ── Step 2: QC indices ───────────────────────────────────────────────
        qc_indices = compute_qc_indices(f, all_channels)
        n_obs = {ch: len(qc_indices[ch]) for ch in channels_used}
        N     = sum(n_obs.values())
        print(f"N obs/channel : {n_obs}")
        print(f"N total       : {N}")

        # ── Step 3: Read HofX for every member ───────────────────────────────
        hofx_all = np.zeros((K, N), dtype=np.float64)
        for k, grp in enumerate(member_groups):
            hofx_all[k] = extract_hofx_vector(f, grp, all_channels, qc_indices)
            if (k + 1) % 50 == 0 or k == K - 1:
                print(f"  read {k+1:4d}/{K} members")

        # ── Step 3b: Read lat/lon for localization ────────────────────────────
        if horizontal_scale is not None:
            lats, lons = extract_latlon_vectors(f, 'MetaData', qc_indices)

    # ── Step 4: HBH^T = 1/(n_ens-1) * X' X'^T ──────────────────────────────
    # Denominator is always (n_ens - 1) regardless of K = n_eig * n_ens,
    # because the modulated ensemble inflates variance by n_eig but does not
    # increase the independent sample count beyond n_ens.
    hofx_mean = hofx_all.mean(axis=0)              # (N,)
    X_pert    = hofx_all - hofx_mean                # (K, N)
    HBHt      = (X_pert.T @ X_pert) / (n_ens - 1)  # (N, N)

    print(f"\nHBH^T shape   : {HBHt.shape}")
    print(f"HBH^T diag    : mean={np.diag(HBHt).mean():.4f}  max={np.diag(HBHt).max():.4f}")

    # ── Step 4b: Apply Gaspari-Cohn localization ─────────────────────────────
    if horizontal_scale is not None:
        print(f"\nLocalization  : horizontal_scale = {horizontal_scale} km  "
              f"(GC → 0 at {2*horizontal_scale} km)")
        L    = build_localization_matrix(lats, lons, horizontal_scale)
        HBHt = L * HBHt
        print(f"Localization matrix : mean={L.mean():.4f}  "
              f"fraction nonzero={np.mean(L > 0):.3f}")
        print(f"HBH^T (localized) diag : mean={np.diag(HBHt).mean():.4f}  "
              f"max={np.diag(HBHt).max():.4f}")

    # ── Step 4c: Diagonal-only option ────────────────────────────────────────
    if diagonal_only:
        HBHt = np.diag(np.diag(HBHt))
        print(f"\ndiagonal_only : off-diagonal elements zeroed out")

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    np.save(output_fname, HBHt)
    meta = {
        'channels_used'         : channels_used,
        'n_obs_per_channel'     : [n_obs[ch] for ch in channels_used],
        'N_total'               : N,
        'K_members'             : K,
        'n_ens'                 : n_ens,
        'denominator'           : n_ens - 1,
        'member_type'           : member_type,
        'member_groups'         : member_groups,
        'localization_scale_km' : horizontal_scale,
        'diagonal_only'         : diagonal_only,
    }
    meta_fname = output_fname.replace('.npy', '_meta.json')
    with open(meta_fname, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved HBH^T   → {output_fname}")
    print(f"Saved metadata → {meta_fname}")


if __name__ == '__main__':
    main()

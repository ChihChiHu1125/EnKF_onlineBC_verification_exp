#!/usr/bin/env python3
'''
code inherited from mpas_jedi_online_satbias_estimator.py

Instead of changing the R matrix (which requires HBH^T might be cumbersome to get the off-diagonal values for high dim Ny)
This leverage the Hdx (from posterior stats) to estimate d beta

Date: 2026/05/14

The update of satbias coefficient & variance for ensemble-DA online BC 
Note: the key difference to the "offline BC estimate" is that the "effective obs error (R_tilde)"
      is now inflated as:
      R_tilde = R + HBH^T
So need variable group name "prior_HofX_variance" available in the obsout_da_ahi-cld_himawari9.h5 file
Date: 2026/04/23

================================
Description: The tool is used to generate the VarBC coefficient and variance for the MPAS-JEDI radiance DA. 
Date: Mar. 2025, initial version

Modified: Mar 2026
(1) bug fix -- Rinv = 1/ (err**2)
(2) add data selection criterion (note: currently the class object is used as a global variable)

'''
import numpy  as np
np.set_printoptions(precision=2)
import h5py
import os, sys, argparse
import yaml

# These globals are set from YAML config in __main__:
qcGroupName       = None
obsGroupName      = None
hofxGroupName     = None
biasGroupName     = None
errGroupName      = None
priorvarGroupName = None
user_defined_predictorNames = None
channels_used     = None
inflation_factor  = None

class DataSelectionCriterion():
    def __init__(self,method, cloud_diff_threshold=0):
        if method not in ['cloud_consistent', 'default']:
            raise ValueError(f"Invalid method {method}; Must be 'cloud_consistent' or 'default' ")
        self.method = method
        self.cloud_diff_threshold = cloud_diff_threshold

# set from YAML config in __main__
dataSelCri = None

class ObsoutReader():
    def __init__(self,fname):
        # db_dict structure: {(groupname,ch):list(values)}.
        # db_dict hold all values read from multi obsout files
        self.db_dict = {}
        # extract predictor and channel information from the first file
        with h5py.File(fname, "r") as f:
            keys  = list(f.keys())
            self.predictorNames   = []
            for key in keys:
                if key.endswith('Predictor'):
                    self.predictorNames.append(key.replace("Predictor",""))
            self.channels = list(f['Channel'][:])
        try:
            user_defined_predictorNames
        except NameError:
            print("Use all predictors in obsout files")
        else:
            self.predictorNames = user_defined_predictorNames
            print("Use user-defined predictors")
        print(self.predictorNames)
    
    def setPredNames(self,predNames):
        self.predictorNames = predNames 
   
    def _readGroupbyName(self,hdf5,groupname):
        # private function
        group   = hdf5[groupname]
        # each group have one variable in obsout, so get the first variable
        varname = group[list(group.keys())[0]].name
        var2d     = hdf5[varname]
        return var2d[:,:] #(location,channel)
    
    def _compute_all_qc_indices(self, hdf5):
        '''Compute QC boolean index for every channel. Returns a list of bool arrays (one per channel).'''
        qc2d = self._readGroupbyName(hdf5, qcGroupName)
        if dataSelCri.method == 'cloud_consistent':
            clrBT   = self._readGroupbyName(hdf5, 'clearBkgBrightnessTemperaturePredictor')
            obsBT   = self._readGroupbyName(hdf5, 'ObsValue')
            modelBT = self._readGroupbyName(hdf5, hofxGroupName) - self._readGroupbyName(hdf5, biasGroupName)
            cld_obs   = np.maximum(clrBT - obsBT, 0)
            cld_model = np.maximum(clrBT - modelBT, 0)
        indices = []
        for ch in self.channels:
            ich = self.channels.index(ch)
            if dataSelCri.method == 'default':
                index = qc2d[:, ich] == 0
            elif dataSelCri.method == 'cloud_consistent':
                index = (qc2d[:, ich] == 0) \
                    & (np.abs(cld_obs[:, ich] - cld_model[:, ich]) < dataSelCri.cloud_diff_threshold) \
                    & (clrBT[:, ich] >= np.maximum(obsBT[:, ich], modelBT[:, ich]))
            indices.append(index)
        return indices

    def _add_var_with_index(self, hdf5, groupname, indices, target_dict):
        '''Read groupname and filter by precomputed QC indices; accumulate into target_dict.'''
        var2d = self._readGroupbyName(hdf5, groupname)
        for ch in self.channels:
            ich      = self.channels.index(ch)
            index    = indices[ich]
            var_qced = var2d[index, ich] if index.any() else np.array([])
            if (groupname, ch) not in target_dict:
                target_dict[(groupname, ch)] = list(var_qced)
            else:
                target_dict[(groupname, ch)].extend(list(var_qced))
    
    def update_tlaps_mean(self,tlapmean_input,tlapmean_output):
        self.tlaps_mean = np.zeros(len(self.channels))
        instrument = ""
        # read in tlaps_mean txt file
        if os.path.exists(tlapmean_input):
            with open(tlapmean_input, "r") as f:
                for line in f:
                    parts = line.split()
                    instrument = parts[0]
                    channel = int(parts[1])
                    value = float(parts[2])
                    if channel in self.channels:
                        ich  = self.channels.index(channel)
                        self.tlaps_mean[ich] = value
        else:
            print("Warnning: tlaps_mean.txt not exist, not update tlaps mean.")
        
        # compute origional tlapse        
        for ch in self.channels:
            ich        = self.channels.index(ch)
            tlapseList = self.db_dict[("lapseRate"+"Predictor",ch)] # list
            tmean      = self.tlaps_mean[ich]
            tlapse_org = np.array(tlapseList) + tmean 
            self.db_dict[("lapseRate"+"Predictor",ch)] = list(tlapse_org)
            self.db_dict[("lapseRate_order_2"+"Predictor",ch)] = list(tlapse_org**2)
        # recompute tlapse mean
        with open(tlapmean_output, "w") as f:
            for ch in self.channels:
                ich        = self.channels.index(ch)      
                tlapseList = self.db_dict[("lapseRate"+"Predictor",ch)] # list      
                tlapse_org = np.array(tlapseList)
                if len(tlapse_org) > 0:
                    tmean  = np.mean(tlapse_org)
                else:
                    # no observation to recompute tlaps mean,keep 0
                    tmean  = 0
                tlapse     = tlapse_org - tmean
                self.db_dict[("lapseRate"+"Predictor",ch)] = list(tlapse)
                self.db_dict[("lapseRate_order_2"+"Predictor",ch)] = list(tlapse**2)
                f.write(f"{instrument} {ch} {tmean:.6E}\n")
            
    def add_prior_file(self, fname):
        '''Read prior obsout file: compute QC, read predictors and R. Returns prior QC indices.'''
        print("read prior  ", fname)
        with h5py.File(fname, "r") as f:
            indices = self._compute_all_qc_indices(f)
            for key in self.predictorNames:
                self._add_var_with_index(f, key+"Predictor", indices, self.db_dict)
            for key in [errGroupName, priorvarGroupName]:
                self._add_var_with_index(f, key, indices, self.db_dict)
        return indices

    def add_posterior_file(self, fname, indices):
        '''Read obs, hofx, bias from posterior obsout file using prior QC indices (for get_Yd only).'''
        print("read posterior", fname)
        with h5py.File(fname, "r") as f:
            for key in [obsGroupName, hofxGroupName, biasGroupName]:
                self._add_var_with_index(f, key, indices, self.db_dict)
    
    def get_channels(self):
        return self.channels
    
    def get_predNames(self):
        return self.predictorNames
    
    def get_Predictor(self,channel):
        res2d = []
        for predName in self.predictorNames:
            key    = (predName+"Predictor",channel)
            values = self.db_dict[key]
            res2d.append(values)
        pred_8xn = np.array(res2d).astype(np.float64)
        return pred_8xn
    
    def get_Yd(self,channel):
        obs = self.db_dict[(obsGroupName,channel)]
        hofx= self.db_dict[(hofxGroupName,channel)]
        bias= self.db_dict[(biasGroupName,channel)]
        # for posterior online bias correction
        # this should be y-H(xa) (without bias correction)
        # i.e., hofx should come from the posterior
        Yd_nx1 = np.array(obs) - np.array(hofx) + np.array(bias)
        Yd_nx1 = Yd_nx1.astype(np.float64)
        return Yd_nx1
    
    def get_Rinv(self, channel):
        err       = np.array(self.db_dict[(errGroupName,     channel)], dtype=np.float64)
        prior_var = np.array(self.db_dict[(priorvarGroupName, channel)], dtype=np.float64)
        Rinv            = 1.0 / err**2
        R_inflated_inv  = 1.0 / (err**2 + prior_var)
        return Rinv, R_inflated_inv

class SatbiasIO():
    def __init__(self,predictorNames,channels_used):
        nch_used = len(channels_used)
        self.predictorNames = predictorNames
        self.channels_used  = channels_used
        self.num_obs_used = np.zeros(nch_used)
        # beta_dict: {'predname':[beta of nChannels]}
        self.beta_dict = {}
        self.err_dict  = {}
        for key in predictorNames:
            self.beta_dict[key] = np.zeros(nch_used, dtype=np.float64)
            self.err_dict[key]  = 1e20*np.ones(nch_used, dtype=np.float64)
    
    def readin_satbias_beta(self,fname,groupname='BiasCoefficients'):
        #inflation_factor = 1.1 -> moved to a global variable
        if os.path.exists(fname):
            with h5py.File(fname, "r") as hdf5:
                file_channels = list(hdf5['Channel'][:])
                group  = hdf5[groupname]
                nch_used = len(self.channels_used)
                for key in list(group.keys()):
                    # initialise entry for predictors in satbias but not in obsout,
                    # so their values carry over rather than being dropped
                    if key not in self.beta_dict and key not in self.err_dict:
                        if groupname == 'BiasCoefficients':
                            self.beta_dict[key] = np.zeros(nch_used, dtype=np.float64)
                            self.err_dict[key]  = 1e20 * np.ones(nch_used, dtype=np.float64)
                        elif groupname == 'BiasCoefficientErrors':
                            if key not in self.err_dict:
                                self.err_dict[key] = 1e20 * np.ones(nch_used, dtype=np.float64)
                            if key not in self.beta_dict:
                                self.beta_dict[key] = np.zeros(nch_used, dtype=np.float64)
                    varname = group[key].name
                    var2d   = hdf5[varname]  #(1,nch_in_file)
                    row     = var2d[0, :]
                    for i, ch in enumerate(self.channels_used):
                        if ch in file_channels:
                            ich_file = file_channels.index(ch)
                            if groupname=='BiasCoefficients':
                                self.beta_dict[key][i] = row[ich_file]
                            elif groupname == 'BiasCoefficientErrors':
                                self.err_dict[key][i]  = row[ich_file]
        else:
            print("Warnning: initial satbias not exist, starting from ZERO beta")
    
    def readin_satbias_err(self,fname,groupname='BiasCoefficientErrors'):
        self.readin_satbias_beta(fname,groupname=groupname)            
    
    def getBeta(self,channel):
        beta = []
        ich  = self.channels_used.index(channel)
        for key in self.predictorNames:
            beta.append(self.beta_dict[key][ich])
        beta = np.array(beta)
        beta_8x1 = beta.T
        return beta_8x1
    
    def setBeta(self,key,channel,value):
        ich  = self.channels_used.index(channel)
        self.beta_dict[key][ich] = value
    
    def getCov(self,channel):
        err = []
        ich  = self.channels_used.index(channel)
        for key in self.predictorNames:
            err.append(self.err_dict[key][ich])
        B = np.diag(err)
        return B 
    
    def setCov(self,key,channel,value):
        ich  = self.channels_used.index(channel)
        self.err_dict[key][ich] = value

    def setNumObsUsed(self,channel,value):
        ich  = self.channels_used.index(channel)
        self.num_obs_used[ich] = value
           
    def _addLabel_to_hdf5Varialbe(self,f,varname,data_dim1,data_dim2):
        h5var = f[varname]
        h5var.dims[0].attach_scale(data_dim1)
        h5var.dims[1].attach_scale(data_dim2)

    def save(self,outname):
        n_Records = 1
        shape     = (n_Records, len(self.channels_used))
        # Create the HDF5 file for writing
        with h5py.File(outname, "w") as f:
            # Set global attributes
            f.attrs["_ioda_layout"] = "ObsGroup"
            f.attrs["_ioda_layout_version"] = 0
            # Create root-level datasets    
            var_channel  = f.create_dataset("Channel", data=np.array(self.channels_used),    dtype=np.int32)       
            var_record   = f.create_dataset("Record",  data=np.array(range(n_Records)), dtype=np.int32)
            # write Number of Observation Used
            num_obs_used = self.num_obs_used.reshape(shape)
            var_num      = f.create_dataset("numberObservationsUsed", data=num_obs_used,dtype=np.int32)      
            self._addLabel_to_hdf5Varialbe(f,'numberObservationsUsed',var_record,var_channel)
            # write for each predictor
            group_beta = f.create_group("BiasCoefficients")       
            group_err  = f.create_group("BiasCoefficientErrors")            
            for key in self.beta_dict.keys():
                beta= self.beta_dict[key].reshape(shape)
                err = self.err_dict[key].reshape(shape)
                ds_beta = group_beta.create_dataset(key,data=beta,dtype=np.float32)
                ds_err  = group_err.create_dataset(key,data=err,dtype=np.float32)
                self._addLabel_to_hdf5Varialbe(f,'BiasCoefficients/'+key, var_record,var_channel)
                self._addLabel_to_hdf5Varialbe(f,'BiasCoefficientErrors/'+key, var_record,var_channel)
        print("Updated satbias to ",outname)  

class MPAS_JEDI_BiasCorrection():
    def __init__(self,obsout_reader,satbias_io):
        self.obsout_reader = obsout_reader
        self.satbias_io    = satbias_io
        self.channels      = obsout_reader.get_channels()
        self.predNames     = obsout_reader.get_predNames() 

    def solve_beta(self,channel):
        # hessian: [B^-1 + P @ R^-1 @ P^T]
        # beta =   [B^-1 + P @ R^-1 @ P^T]^-1 [B^-1@beta0 + P @ R^-1 @ Yd]
        pred_8xn = self.obsout_reader.get_Predictor(channel)
        Rinv, Rinfinv  = self.obsout_reader.get_Rinv(channel)
        Yd_nx1   = self.obsout_reader.get_Yd(channel)
        beta     = self.satbias_io.getBeta(channel)
        B        = inflation_factor * self.satbias_io.getCov(channel)        
        #Binv     = np.linalg.inv(B)
        Binv     = np.linalg.pinv(B)
        #self.print_residual("BEFORE:",channel,pred_8xn,beta,Yd_nx1)        
        # solve beta
        P_Rinv    = pred_8xn * Rinv[None, :]  #pred_8xn @ Rinv
        P_Rinfinv = pred_8xn * Rinfinv[None, :]  #pred_8xn @ Rinv

        P_Rinv_Pt = P_Rinv @ pred_8xn.T
        P_Rinfinv_Pt = P_Rinfinv @ pred_8xn.T

        hessian   = Binv + P_Rinv_Pt
        hessian_inf = Binv + P_Rinfinv_Pt

        A         = np.linalg.pinv(hessian)
        A_inf     = np.linalg.pinv(hessian_inf)

        term2 = Binv @ beta + P_Rinv @ Yd_nx1        
        beta  = A @ term2        
        for i in range(len(self.predNames)):
            key = self.predNames[i]
            self.satbias_io.setBeta(key,channel,beta[i])
            #self.satbias_io.setCov (key,channel,A[i,i])
            self.satbias_io.setCov (key,channel,A_inf[i,i])  # use the analytical solution for analysis variance

        self.satbias_io.setNumObsUsed (channel,len(Yd_nx1))        
        self.beta = beta
        self.print_residual("UPDATE:",channel,pred_8xn,beta,Yd_nx1,A,A_inf)
        #print("-----------------------")
        return beta,hessian
    
    def print_residual(self,prefix,channel,pred_8xn,beta,Yd_nx1,A,A_inf):        
        shape   = pred_8xn.shape
        nObs    = shape[1]
        #if nObs == 0:
        #    return
        obsbias = pred_8xn.T @ beta
        omb_bc  = Yd_nx1 - obsbias
        string  = "%s ch%d "%(prefix,channel)
        obsbias_components = [];
        if nObs>0:
            string += " RMS(omb_bc)= %.2f "%np.sqrt(np.mean(omb_bc**2))
            string += " MEAN(omb_bc)= %8.4f "%np.mean(omb_bc)
            string += " obsbias= %.2f ["%np.mean(obsbias)
            for i in range(len(self.predNames)):
                key   = self.predNames[i]
                bias  = pred_8xn.T[:,i:i+1] @ beta[i:i+1] # beta[i:i+1] is the ith beta value
                string += " %.2f "%(np.mean(bias))
                obsbias_components.append(np.mean(bias))
            string += "]"
        else:
            string += " no valid observations"
        print(string)

        # record these numbers, and will be save into .csv file
        record[channel]={};
        record[channel]['obs_num'] = nObs
        record[channel]['bias_total'] = np.mean(obsbias)
        record[channel]['bias_components'] = obsbias_components
        record[channel]['beta'] = beta
        #record[channel]['beta_variance'] = np.diag(A)  
        record[channel]['beta_variance'] = np.diag(A_inf)  


    def update(self):
        for ch in channels_used:
            self.solve_beta(ch)       

    def save(self,outname):
        self.satbias_io.save(outname)     

# function to save the record (print info) into a .csv file:
def save_record_into_csv(output_filename):
    max_len = max(len(v["bias_components"]) for v in record.values())
    rows = []
    for k, v in record.items():
        comps = list(v["bias_components"])
        beta = list(v["beta"])
        beta_variance = list(v["beta_variance"])
        
        row = {
            "channel": int(k),
            "obs_num": v["obs_num"],
            "bias_total": v["bias_total"],
        }
        for i in range(max_len):
            if i < len(comps):
                row[f"bias_{user_defined_predictorNames[i]}"] = comps[i]
                row[f"beta_{user_defined_predictorNames[i]}"] = beta[i]
                row[f"beta_variance_{user_defined_predictorNames[i]}"] = beta_variance[i]
            else:
                row[f"bias_{user_defined_predictorNames[i]}"] = np.nan
                row[f"beta_{user_defined_predictorNames[i]}"] = np.nan
                row[f"beta_variance_{user_defined_predictorNames[i]}"] = np.nan
        
        rows.append(row)
    
    df_long = pd.DataFrame(rows)
    df_long.to_csv(output_filename, index=False)



# =========================================================

record={};

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='mpas_jedi_online_satbias_estimator.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # set global variables from config
    qcGroupName       = cfg['group_names']['qc']
    obsGroupName      = cfg['group_names']['obs']
    hofxGroupName     = cfg['group_names']['hofx']
    biasGroupName     = cfg['group_names']['bias']
    errGroupName      = cfg['group_names']['obs_error']
    priorvarGroupName = cfg['group_names']['prior_var']
    user_defined_predictorNames = cfg['predictors']
    channels_used     = cfg['channels_used']
    inflation_factor  = cfg['inflation_factor']
    dataSelCri = DataSelectionCriterion(
        cfg['data_selection']['method'],
        cfg['data_selection']['cloud_diff_threshold']
    )

    obsout_posterior_fnames = cfg['files']['obsout_posterior']
    obsout_prior_fnames     = cfg['files']['obsout_prior']
    satbias_input  = cfg['files']['satbias_input']
    satbias_output = cfg['files']['satbias_output']
    csv_output     = cfg['files'].get('csv_output', '')

    if len(obsout_posterior_fnames) != len(obsout_prior_fnames):
        raise ValueError("obsout_posterior and obsout_prior must have the same number of files")

    print(obsout_prior_fnames[0])

    #=================================
    # run the estimator
    obsout = ObsoutReader(obsout_prior_fnames[0])
    for prior_fname, post_fname in zip(obsout_prior_fnames, obsout_posterior_fnames):
        qc_indices = obsout.add_prior_file(prior_fname)
        obsout.add_posterior_file(post_fname, qc_indices)
    predNames = obsout.get_predNames()
    channels  = obsout.get_channels()
    satbias   = SatbiasIO(predNames, channels_used)  # note: only used channels_used for BC
    satbias.readin_satbias_beta(satbias_input)
    satbias.readin_satbias_err(satbias_input)
    bc = MPAS_JEDI_BiasCorrection(obsout, satbias)
    bc.update()
    bc.save(satbias_output)

    # save the print info into .csv file:
    if csv_output:
        import pandas as pd
        save_record_into_csv(csv_output)

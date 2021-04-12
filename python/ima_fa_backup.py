import datareader, datafilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn import model_selection
from scipy import signal, stats
from numba import cuda
import time
import aopy

# Generic Functions
def ima_preprocess(data_in, time_start, time_stop, srate, on, ftype, order, Wn, verbose = True, var_thresh = False):
    '''
    This function preprocesses data before dimensionality reduction by subsampling the data for a given time window.
    This function will return 'None' if the variance is above threshold
    
    Inputs:
        data_in [nchannel, ntime]: Raw data in
        time_start [int]: Time to start analysis (s)
        time_stop [int]: Time to stop analysis (s)
        srate [int]: Sampling rate (hz)
        on [bool]: If filtering is turned on
        ftype [string]: Type of filter (bandpass, lowpass, highpass)
        order [int]: Filter order
        Wn [tuple]: Cut off frequencies
        verbose [bool]: Display start and stop indices
        var_thresh [int]: Maximum variance threshold. 
        
    Output:
        data_out [nchannel, ntime]: 2D np array, subsampled and filtered data
    '''
    
    # Define subsampled time range
    if type(time_start) != int or type(time_stop) != int:
        stidx = (time_start*srate).astype(int)
        spidx = (time_stop*srate).astype(int)
    else:
        stidx = time_start*srate
        spidx = time_stop*srate
    
    # Check that data doesn't contain any artifacts if a max variance value is set
    if type(var_thresh) is int:
        data_in_var = np.var(data_in[:,stidx:spidx])
        if data_in_var > var_thresh:
            return None
        
    # Print start and stop index of the subsample analyzed.
    if verbose == True:
        print('Start Index: ' + str(stidx))
        print('Stop Index: ' + str(spidx))
    data = data_in[:,stidx:spidx]
    
    # Butterworth filter
    if on is True:
        sos = signal.butter(order, Wn, ftype, fs = srate, output = 'sos')
        data_out = signal.sosfiltfilt(sos, data, 1)    
    
    return data_out

def ima_fa(data_in, ncomp_fa, nfold, maxiter, verbose):
    '''
    Apply cross validated factor analysis (FA) to input data. Cross validated scores are averaged across n folds.
    
    Inputs:
        data_in [nchannel, ntime]: Data in
        ncomp_fa [n dimensions]: Array of dimensions to compute FA across 
        nfold [int]: Number of cross validation folds to compute
        maxiter [int]: Maximum number of FA iterations to compute if there is no convergence
        verbose [bool]: Display % of dimensions completed
    Outputs:
        fascore['test_score_mu'] [n dimensions]: Array of MLE FA score for each dimension
        fascore['converge'] [n dimensions]: Indicates if the FA converged before reaching the maximum iteration value.
                0: FA did not converge
                1: FA converged
    '''
    
    # Initialize arrays
    fascore = {'test_scores':  np.zeros((np.max(np.shape(ncomp_fa)), nfold))}
    fascore['test_score_mu'] = np.zeros((np.max(np.shape(ncomp_fa))))
    fascore['converge'] = np.zeros(np.max(np.shape(ncomp_fa)))
    converge = np.zeros((nfold))
    
    # Split data into training and testing sets and perform FA analysis for each dimension
    if verbose == True:
        print('Cross validating and fitting ...')
    
    for jj in range(len(ncomp_fa)):
        ii = 0
        for trainidx, testidx in model_selection.KFold(n_splits = nfold).split(data_in):

            fa = FactorAnalysis(n_components = ncomp_fa[jj], max_iter = maxiter)
            fafit = fa.fit(data_in[trainidx,:])
            fascore['test_scores'][jj,ii] = fafit.score(data_in[testidx,:])
            converge[ii] = fafit.n_iter_
            ii += 1
        
        fascore['test_score_mu'][jj] = np.mean(fascore['test_scores'][jj,:])
        
        # Return flag if the factor analysis doesn't converge
        if max(converge) == maxiter:
            fascore['converge'][jj] = 0
        else:
            fascore['converge'][jj] = 1
        
        if verbose == True:
            print(str((100*(jj+1))//len(ncomp_fa)) + "% Complete")
            
    return fascore

def ima_combine_results(p2l1, p2l2, p2s):
    '''
    Concatenate results from multiple FA result files and save to a file
    
    Input:
        p2l1 [list of strings]: List of file paths to iterate through
        p2l2 [list of strings]: List of file paths to iterate through to combine with p2l2
        p2s [list of strings]: List of file paths to save concatenated arrays to
        
    Output:
        No returned outputs. This function saves the concatenated array for each filepath combination to the corresponding 
        p2s filepath.
    '''
    
    # Check if all lists are the same length
    if len(p2l1) != len(p2l2) != len(p2s):
        raise 
        
    for ii in range(len(p2l1)):
        data1 = np.load(p2l1[ii])
        data2 = np.load(p2l2[ii])

        data_out = np.concatenate((data1, data2), axis = 2)
        np.save(p2s[ii], data_out)
        
def ima_setvarThresh(p2l, var_multiplier):
    '''
    Determine variance threshold by calculating the variance of the entire dataset and multiplying it by a specified number
    
    Input:
        p2l [string]: Path to load raw data set
        var_multiplier [int]: threshold multiplier to determine a variance cutoff
        
    Output:
        thresh [float]: Variance threshold
    '''
    
    # Load data set and determine 3 std 
    data_in, data_param, data_mask = datareader.load_ecog_clfp_data(p2l)
    thresh = var_multiplier*np.var(data_in)
    return thresh

def plot_FAdims(data, state, save_figs, p2s, vmaxin = 20, cblbl = 'Dimensionality', xlbl = 'Trial Number', extent_bnds = None):
    '''
    Plot factor analysis (FA) results as an image across multiple trials
    
    Inputs:
        data [n freq band, ntrials]: FA results across multiple trials
        state [string]: Figure title
        save_figs [bool]: Save figures or not
        p2s [string]: Path to save results
        vmaxin [int]: Max colormap c axis value
        cblbl [string]: Colormap label
        xlbl [string]: X axis lbl
        extent_bnds [xmin, xmax, ymax, ymin]: Range of image shown
        
    Outputs:
        No returned outputs. Shows a figure with dimensionality results across trials and saves to a specified filepath if desired
    '''
    f,ax = plt.subplots(1,1,figsize=(10,4))
    plt.imshow(data, aspect = 'auto', vmin = 0, vmax = vmaxin, extent = extent_bnds)
    cb = plt.colorbar(fraction = 0.03, pad = 0.01)
    cb.ax.set_ylabel(cblbl, fontsize = 24, rotation = 270, labelpad=25)
    cb.ax.tick_params(labelsize=20)
    plt.xlabel(xlbl, fontsize = 24)
    plt.ylabel('Freq Band Number', fontsize = 24)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title(state, fontsize = 28)
    if save_figs == True:
        f.savefig(p2s,bbox_inches = "tight", dpi = 300)

def plot_FAhist(data, state, save_figs, p2s):
    '''
    This function plots a histogram showing the distribution of dimensionality for each frequency band given multiple trials
    
    Inputs:
        data [n freq band, ntrials]: FA results across multiple trials
        state [string]: Figure title
        save_figs [bool]: Save figures or not
        p2s [string]: Path to save results
        
    Outputs:
        No returned output. Shows a figure and saves the figure if desired.
    '''
    
    fig = plt.figure(figsize = (10,4))
    plt.title(state, fontsize = 28)
    plt.xlabel('Dimensionality', fontsize = 20)
    plt.ylabel('Count', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    for ii in range(data.shape[0]):
        plt.hist(data[ii,:], bins = 5, alpha = 0.5, label=str(ii))

    plt.legend(prop = {'size': 16})
    plt.show()
    if save_figs == True:
        fig.savefig(p2s, bbox_inches = "tight", dpi = 300)
        
def ima_load_FAresults(p2l, minFAdim = 1):
    '''
    Load maximum likelihood estimates from factor analysis (FA) results
    
    Inputs:
        p2l [list of strings]: List of .npy datapath to load FA results
        minFAdim [int]: Minimum FA dimension analyzed
        
    Output:
        fa_out [n freq band, ntrial, npath]: Dimension with maximum likelihood value for each frequency band, trial and path
    '''
    
    # Load first data set to get data shape
    temp = np.load(p2l[0])
    
    # Initialize variables
    fa_out = np.zeros((temp.shape[0], temp.shape[2], len(p2l)))
    
    for ii in range(len(p2l)):
        fa_mle = np.load(p2l[ii])
        fa_out[:,:,ii] = np.argmax(fa_mle, 1) + minFAdim
    
    return fa_out


def ima_ptest(ptest_data):
    '''
    Calculate ptest results comparing all combinations of each frequency band input
    
    Inputs:
        ptest_data [n freq band, ntrials, nstate]: Data set containing the dimensionality for each trial across each
        frequency band and neural state to test
        
    Outputs:
        ptest_results [nstate, nstate, n freq band]: pscore results comparing the neural states for each frequency band
    '''
    
    ptest_results = np.zeros((ptest_data.shape[2], ptest_data.shape[2], ptest_data.shape[0]))
    for ii in range(ptest_data.shape[0]):
        for jj in itertools.combinations_with_replacement([0,1,2,3], 2):
            ttest = stats.ttest_ind(ptest_data[ii,:,jj[1]], ptest_data[ii,:,jj[0]], equal_var = False)
            ptest_results[jj[1], jj[0], ii] = ttest.pvalue
    return ptest_results

# Plotting functions specific to the tests I ran
def ima_loadFAplot(p2l, p2s, save_figs, minFAdim = 1):
    '''
    Loads save FA results, calculates dimensionality, and plots dimensionality across trials and a histogram of 
    dimensionality distribution across trials. Also prints the mean and standard deviation of dimensionality across
    the trials.
    
    Inputs:
        p2l [string]: File path to load FA results
        p2s [string]: File path to save figures
        save_figs [bool]: Save figures or not
        minFAdim [int]: Starting dimension FA is performed on
        
    Output
        No returned outputs. Creates and saves images and prints average dimensionality and standard deviation.
    '''
    
    fa_mle = np.load(p2l)

    fa_max_mle = np.argmax(fa_mle, 1) + minFAdim

    # Plot dimensionality for asleep data
    plot_FAdims(fa_max_mle, 'Asleep', save_figs, p2s + '.png')

    # Plot Histogram
    plot_FAhist(fa_max_mle, 'Asleep', save_figs, p2s + '_hist.png')

    # Calculate Mean and Std
    sleep_mu = np.mean(fa_max_mle, axis = 1)
    sleep_std = np.std(fa_max_mle, axis = 1)
    print('1-4 Hz Avg/Std Dimensionality: ', round(sleep_mu[0],1), '/', round(sleep_std[0],1))
    print('4-10 Hz Avg/Std Dimensionality: ', round(sleep_mu[1],1), '/', round(sleep_std[1],1))
    print('10-20 Hz Avg/Std Dimensionality: ', round(sleep_mu[2],1), '/', round(sleep_std[2],1))
    print('20-40 Hz Avg/Std Dimensionality: ', round(sleep_mu[3],1), '/', round(sleep_std[3],1))
    print('40-70 Hz Avg/Std Dimensionality: ', round(sleep_mu[4],1), '/', round(sleep_std[4],1))
    print('70-100 Hz Avg/Std Dimensionality: ', round(sleep_mu[5],1), '/', round(sleep_std[5],1))
    print('\n')
    
def ima_plot_cummu(data,fband, titlelbl, p2s,save_figs):
    '''
    This function plots the results of calculating the cumulative average across trials for each predefined state
    
    Inputs:
        data [nfreq band, navged trial, nstate]: Cumulative average data for 50 trials and 4 specific states
        fband [int]: Which frequency band to analyze
        titlelbl [string]: ID to name figure and saved figure. Typically the frequency band analyzed
        p2s [string]: Path to save data
        save_figs [bool]: Save figures or not
        
    Outputs:
        No defined outputs. Creates a figure and saves it
    '''
    fig = plt.figure(figsize = (10,6))
    plt.plot(np.arange(1, 51),data[fband,:,0], label = 'Spont Rec002', linewidth = 2)
    plt.plot(np.arange(1, 51),data[fband,:,1], label = 'Spont Rec004', linewidth = 2)
    plt.plot(np.arange(1, 51),data[fband,:,2], label = 'VisEP Rec003', linewidth = 2)
    plt.plot(np.arange(1, 51),data[fband,:,3], label = 'Task Reck 007', linewidth = 2)
    plt.xlabel('Number of Trials Averaged', fontsize = 18)
    plt.ylabel('Dimensionality', fontsize = 18)
    plt.title(titlelbl, fontsize = 22)
    plt.ylim((0, 8.5))
    plt.legend(bbox_to_anchor=(1, 1), fontsize = 12)
    plt.show()
    if save_figs == True:
        fig.savefig('E:/aolab/codes/ima/python/Results/180325/cummu_'+titlelbl, bbox_inches = "tight", dpi = 300)
        
def ima_plot_cumSTD(data,fband, titlelbl, p2s,save_figs):
    '''
    This function plots the results of calculating the cumulative standard deviation across trials for each predefined state
    
    Inputs:
        data [nfreq band, navged trial, nstate]: Cumulative standard deviation data for 50 trials and 4 specific states
        fband [int]: Which frequency band to analyze
        titlelbl [string]: ID to name figure and saved figure. Typically the frequency band analyzed
        p2s [string]: Path to save data
        save_figs [bool]: Save figures or not
        
    Outputs:
        No defined outputs. Creates a figure and saves it
    '''
    fig = plt.figure(figsize = (10,6))
    plt.plot(np.arange(1, 51),data[fband,:,0], label = 'Spont Rec002', linewidth = 2)
    plt.plot(np.arange(1, 51),data[fband,:,1], label = 'Spont Rec004', linewidth = 2)
    plt.plot(np.arange(1, 51),data[fband,:,2], label = 'VisEP Rec003', linewidth = 2)
    plt.plot(np.arange(1, 51),data[fband,:,3], label = 'Task Reck 007', linewidth = 2)
    plt.xlabel('Number of Trials Averaged', fontsize = 18)
    plt.ylabel('Dimensionality', fontsize = 18)
    plt.title(titlelbl, fontsize = 22)
    plt.ylim((0, 8.5))
    plt.legend(bbox_to_anchor=(1, 1), fontsize = 12)
    plt.show()
    if save_figs == True:
        fig.savefig('E:/aolab/codes/ima/python/Results/180325/cumstd_'+titlelbl, bbox_inches = "tight", dpi = 300)  

def ima_plot_ptest(data, titlelbl, annotate, save_figs, p2s = ''):
    '''
    This function creates an images showing the ptest results and saves the figure
    
    Inputs:
        data [nstate, nstate]: Data with pscores between individual neural states
        titlelbl [string]: ID to name figure and save plot 
        annotate [bool]: To print the pscore for each combination on the figure
        save_figs [bool]: Save figures or not
        p2s [string]: Path to save data
    '''
    
    f,ax = plt.subplots(1,1,figsize=(10,4))
    plt.imshow(data, aspect = 'auto', vmin = 0, vmax = 1)
    cb = plt.colorbar(fraction = 0.03, pad = 0.01)
    cb.ax.set_ylabel('pscore', fontsize = 24, rotation = 270, labelpad=25)
    cb.ax.tick_params(labelsize=20)
    plt.xlabel('Neural state ID', fontsize = 24)
    plt.ylabel('Neural state ID', fontsize = 24)
    plt.xticks(ticks = [0,1, 2, 3], fontsize = 20)
    plt.yticks(ticks = [0,1, 2, 3], fontsize = 20)
    plt.title(titlelbl, fontsize = 28)
    
    # Add labels to each box
    if annotate == True:
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                t = ax.annotate(str(round(data[ii,jj], 3)), (jj-.175,ii+.1), xycoords = 'data', fontsize = 20, color = 'Gray')
    
    if save_figs == True:
        f.savefig((p2s+titlelbl),bbox_inches = "tight", dpi = 600)
        print((p2s+titlelbl))

# Full processing wrappers specific to my tests
def fullProcessing(datapath, p2s, iterations, dtime, freq_bands, var_thresh = False, data_limit_time = None, data_limit_chan = None, max_dims = 21):
    '''
    This function is a wrapper to run factor analysis with artifact detection with the option to crop the data set 
    in time or by channels.
    
    Inputs:
        datapath [list of strings]: List of strings to fun FA on 
        p2s [list of strings]: List or strings to save FA data in (.npy)
        iterations [int]: Number of trials to run for each datapath
        dtime [int]: Time length to cut each trial to [s]
        freq_bands [n bands, 2]: numpy array of frequncy bands
        var_thresh [int]: Variance threshold for outlier detection
        data_limit_time [start, stop]: Pre limit data to select trials from in time
        data_limit_chan [nchannelID]: Pre limit data to select trials from the channel indices input
        max_dims [int]: maximum number of dimensions to run FA on
        
    Output:
        fa_mle [n freq bands, ndimension, niterations]: FA log likelihood results for each trial at each frequency band
    '''
    # Initialize dictionary
    proc_param = {'time_start': 0, 'time_stop': 0} 
    fa_param = {'ncomp_fa': np.arange(1, max_dims ,1).astype(int), 'nfold': 4, 'maxiter':50000}

    # Load data
    data_in, data_param, data_mask = datareader.load_ecog_clfp_data(datapath)
    
    # Parse data if necessary
    if data_limit_time is not None:
        data_in = data_in[:,(data_limit_time[0]*data_param['srate']):(data_limit_time[1]*data_param['srate'])]
    if data_limit_chan is not None:
        data_in = data_in[data_limit_chan,:]
    
    print('Raw data shape:', data_in.shape)
    
    # Randomly select time ranges
    time_start_rng = np.zeros(iterations)
    time_start_rng = np.random.randint(1,(data_in.shape[1]-dtime)//data_param['srate'], size=iterations)
    
    # Define time stop range
    time_stop_rng = time_start_rng + dtime #[s]
    
    # Check if there are artifacts
    if type(var_thresh) is int:
        for nn in range(len(time_start_rng)):
            temp_data_var = np.var(data_in[:,(time_start_rng[nn]*data_param['srate']):((time_start_rng[nn]+dtime)*data_param['srate'])])
            while temp_data_var > var_thresh:
                print(temp_data_var)
                time_start_rng[nn] = np.random.randint(1,(data_in.shape[1]-dtime)//data_param['srate'], size=1)
                temp_data_var = np.var(data_in[:,(time_start_rng[nn]*data_param['srate']):((time_start_rng[nn]+dtime)*data_param['srate'])])
    
    # Preallocate data arrays
    fa_mle = np.zeros((freq_bands.shape[0], fa_param['ncomp_fa'].shape[0], len(time_start_rng)))
    fa_niter = np.zeros((fa_mle.shape))

    a = time.time()
    # Process data for each start point
    for jj in range(len(time_start_rng)):
        proc_param['time_start'] = time_start_rng[jj]
        proc_param['time_stop'] = time_stop_rng[jj]

        # Compute factor analysis for each frequency band
        for ii in range(freq_bands.shape[0]):
            print('f band ', ii)
            filt_param = {'on': True, 'order': 4, 'ftype': 'bandpass', 'Wn': freq_bands[ii,:]}

            # Process data for each frequency band
            data_proc = ima_preprocess(data_in, **data_param, **filt_param, **proc_param, verbose = False)

            # Compute FA
            fa_out = ima_fa(data_proc, **fa_param, verbose = False)

            # Store MLE
            fa_mle[ii, :, jj] = fa_out['test_score_mu'][:]
            fa_niter[ii,:,jj] = fa_out['converge'][:]

            # Clear fa_out to save memory
            del fa_out

        print(str(100*(jj+1)/len(time_start_rng)) + '% Done')

    b = time.time()
    np.save(p2s, fa_mle)
    print('total fa processing time: ' + str(round(b-a,2)) + 'seconds')
    return fa_mle


def fullPlot(data, p2s, save_figs, minFAdim =1):   
    '''
    Plot the dimensionality after each state runs
    
    Inputs:
        data [n freq band, n dimensions, ntrials]: FA results from full processing
        p2s [list of strings]: File paths to saves figures
        save_figs [bool]: Save figures or not
        minFAdim [int]: Start of dimensions analyzed
        
    Output:
        No returned output. Saves figures and print the average and standard deviation of the dimensionality.
    '''
    fa_max_mle = np.argmax(data, 1) + minFAdim 
    fa_max_mle.shape

    # Plot Dimensionality
    plot_FAdims(fa_max_mle, 'Dimensionality', save_figs, p2s + '.png')

    # Plot Histogram
    plot_FAhist(fa_max_mle, 'Dimensionality Distribution', save_figs, p2s + 'hist.png')

    # Calculate Mean and Std
    awake_mu = np.mean(fa_max_mle, axis = 1)
    awake_std = np.std(fa_max_mle, axis = 1)
    print('1-4 Hz Avg/Std Dimensionality: ', round(awake_mu[0],1), '/', round(awake_std[0],1))
    print('4-10 Hz Avg/Std Dimensionality: ', round(awake_mu[1],1), '/', round(awake_std[1],1))
    print('10-20 Hz Avg/Std Dimensionality: ', round(awake_mu[2],1), '/', round(awake_std[2],1))
    print('20-40 Hz Avg/Std Dimensionality: ', round(awake_mu[3],1), '/', round(awake_std[3],1))
    print('40-70 Hz Avg/Std Dimensionality: ', round(awake_mu[4],1), '/', round(awake_std[4],1))
    print('70-100 Hz Avg/Std Dimensionality: ', round(awake_mu[5],1), '/', round(awake_std[5],1))
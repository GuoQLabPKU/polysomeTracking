import pandas as pd
import numpy as np
import scipy
import scipy.stats
from scipy.stats import chi2
import matplotlib.pyplot as plt


def tom_fitDist(inputData, distModel, clusterClass = '',saveDir = '', verbose = 0):
    '''
    TOM_FITDIST is aimed to return the best fitted distribution 
    of input 1D-array
    
    EXAMPLE
    transList = tom_addTailRibo(np.random.random(size=100), ['norm','gamma']);
    
    PARAMETERS
    
    INPUT
        inputData        1D-array to fit specific distribution model        
        distModel        distribution model to fit. Could be one distribution like norm,
                         or multiply distributions like ['norm','gamma']. Five 
                         common distribution models are offered.['expon',
                         'gamma', 'lognorm', 'norm']
        clusterClass     ('',opt) the cluster class of transformations
        saveDir          ('',opt) the pathway to save the summary figures
                         
    OUTPUT
        summaryData     (dataframe) information of  fitted model names, fitted params,
                        KS-test(test the goodness of fitted model) statics and p-values,
                        X^2 test statics and p-value(test the goodness of fitted model)   
    Reference
        for how to manually operate Chi-square test, read this:http://www.stat.yale.edu/Courses/1997-98/101/chigf.htm
        Be careful with the degree of freedom: 
        number of bins - number of parameters fit (normal distribution is 2) -1
    '''
    #check the input data 
    dist_names = [
                  'expon',
                  'gamma',
                  'lognorm',
                  'norm']
    if isinstance(distModel, str):
         if distModel not in dist_names:
             raise TypeError('Not a recognized distribution model, \
                             only %s are permitted.'%str(dist_names))
         distModel = [distModel]
    if isinstance(distModel, list):
        if len(set(distModel) - set(dist_names)) > 0:
             raise TypeError('Not recognized distribution models, \
                             only %s are permitted.'%str(dist_names))            
        
    
    # Set up empty lists to store results
    chi_square = []
    chi_pValue = []
    KS_stat = []
    KS_pValue = []
    fit_params = [ ]
    fit_paramsSave = [ ]
    size = len(inputData)
    #normalize the inputData with mean:0 and std:1
    norData = normalizeData(inputData)
    # Set up 20 bins for chi-square test, so that each bin can have more than 5 samples
    # Observed data will be approximately evenly distrubuted aross all bins
    binN = int(np.ceil(len(norData)/5))
    binN = np.min((binN,21))
    percentile_bins = np.linspace(0,100,binN)
    percentile_cutoffs = np.percentile(norData, percentile_bins)
    observed_number, bins = (np.histogram(norData, bins=percentile_cutoffs))  
    # Loop through candidate distributions   
    for distribution in distModel:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(norData);fit_paramsSave.append(str(param));fit_params.append(param)
        dof = 20-len(param)-1#freedom degree of chi square-test        
        # Obtain the KS test P statistic, round it to 5 decimal places
        stat, p = scipy.stats.kstest(norData, distribution, args=param)
        p = np.around(p, 3)
        KS_pValue.append(p) 
        KS_stat.append(np.round(stat,2))
        # Get expected counts in percentile bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                              scale=param[-1])
        expected_num = []
        for bins in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bins+1] - cdf_fitted[bins]
            expected_num.append(expected_cdf_area)
        
        # calculate chi-squared
        expected_num = np.array(expected_num) * size
        ss = np.sum(((expected_num - observed_number) ** 2) / observed_number)
        chi_square.append(np.round(ss,2))
        pval = 1 - chi2.cdf(ss, dof)
        chi_pValue.append(np.round(pval,3))
    # Collate results and sort by goodness of fit (best at top) 
    results = pd.DataFrame()
    results['distribution'] = distModel
    results['chi_square'] = chi_square
    results['chi_pValue'] = chi_pValue
    results['KS_stat'] = KS_stat
    results['KS_pValue'] = KS_pValue
    results['fit_params'] = fit_paramsSave
    results['fit_params2'] = fit_params
    results.sort_values(['chi_square'], inplace=True)       
    # Report results    
    if verbose:
        print ('\nDistributions sorted by goodness of fit:')
        print ('----------------------------------------')
        print (results)
    #save the data 
    results = results[results['KS_pValue'] > 0.05]
    #plot the figures of distance and the fitted line
    if results.shape[0] > 0:
        createFitPlot(norData, results['distribution'].values, 
                      results['fit_params2'].values,
                      clusterClass, saveDir)
        if len(saveDir) > 0:
            results.drop(['fit_params2'],inplace = True,axis = 1)
            results.to_csv('%s/distFit_c%d.csv'%(saveDir, clusterClass), sep = ",",index = False)
            
    
 

def createFitPlot(inputData, dist_names, fit_params,clusterClass, saveDir):  
    '''
    plot the fittted line
    
    '''  
    # Divide the observed data into 20 bins for plotting 
    number_of_bins = 20
    x = np.linspace(np.percentile(inputData,0),np.percentile(inputData,99),len(inputData))
    bin_cutoffs = np.linspace(np.percentile(inputData,0), np.percentile(inputData,99),number_of_bins)
    # Create the plot
    h = plt.hist(inputData, bins = bin_cutoffs, color='0.75')
    # Loop through the distributions ot get line fit and paraemters  
    for dist_name,param in zip(dist_names, fit_params):
        dist = getattr(scipy.stats, dist_name)
        # Get line for each distribution (and scale to match observed data)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf        
        # Add the line to the plot
        plt.plot(x,pdf_fitted, label=dist_name)
        # Set the plot x axis to contain 99% of the data
        # This can be removed, but sometimes outlier data makes the plot less clear
        plt.xlim(np.percentile(inputData,0),np.percentile(inputData,99))    
    # Add legend and display plot   
    plt.legend(fontsize = 15)
    plt.xlabel('Normalized distance between\neach transformation and Tavg',fontsize = 15)
    plt.ylabel('# of transformation',fontsize = 15)
    plt.tight_layout()
    if len(saveDir) > 0:
        plt.savefig('%s/c%d_fitDis.png'%(saveDir, clusterClass), dpi = 300)
    plt.show()
    #plt.close()
    createQQ_PP(inputData, dist_names, fit_params, clusterClass, saveDir)

def createQQ_PP(inputData, distName, fitParams, clusterClass, saveDir):
    '''
    create QQ plot and PP plot figures
    '''
    data = inputData.copy()
    data.sort()

    for distribution, param in zip(distName, fitParams):
        # Set up distribution
        dist = getattr(scipy.stats, distribution)
        # Get random numbers from distribution
        norm = dist.rvs(*param[0:-2],loc=param[-2], scale=param[-1],size = len(inputData))
        norm.sort()      
        # Create figure
        fig = plt.figure(figsize=(8,5))       
        # qq plot
        ax1 = fig.add_subplot(121) # Grid of 2x2, this is suplot 1
        ax1.plot(norm,data,"o")
        min_value = np.floor(min(min(norm),min(data)))
        max_value = np.ceil(max(max(norm),max(data)))
        ax1.plot([min_value,max_value],[min_value,max_value],'r--')
        ax1.set_xlim(min_value,max_value)
        ax1.set_xlabel('Theoretical quantiles',fontsize = 15)
        ax1.set_ylabel('Observed quantiles',fontsize = 15)
        title = 'qq plot for ' + distribution +' distribution'
        ax1.set_title(title, fontsize = 15)
        
        # pp plot
        ax2 = fig.add_subplot(122)
        
        # Calculate cumulative distributions
        bins = np.percentile(norm,range(0,101))
        data_counts, bins = np.histogram(data,bins)
        norm_counts, bins = np.histogram(norm,bins)
        cum_data = np.cumsum(data_counts)
        cum_norm = np.cumsum(norm_counts)
        cum_data = cum_data / max(cum_data)
        cum_norm = cum_norm / max(cum_norm)
        
        # plot
        ax2.plot(cum_norm,cum_data,"o")
        min_value = np.floor(min(min(cum_norm),min(cum_data)))
        max_value = np.ceil(max(max(cum_norm),max(cum_data)))
        ax2.plot([min_value,max_value],[min_value,max_value],'r--')
        ax2.set_xlim(min_value,max_value)
        ax2.set_xlabel('Theoretical cumulative distribution', fontsize = 15)
        ax2.set_ylabel('Observed cumulative distribution', fontsize = 15)
        title = 'pp plot for ' + distribution +' distribution'
        ax2.set_title(title, fontsize = 15)
        
        # Display plot    
        plt.tight_layout(pad=4)
        if len(saveDir) > 0:
            plt.savefig('%s/c%d%s_QP.png'%(saveDir, clusterClass, distribution), dpi = 300)
        plt.show()
        #plt.close()
    
def normalizeData(inputData):
    inputMean = np.mean(inputData)
    inputStd = np.std(inputData)
    norData = (inputData - inputMean)/inputStd
    return norData
    
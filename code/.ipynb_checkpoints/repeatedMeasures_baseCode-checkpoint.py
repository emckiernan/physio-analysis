# import computing modules
import csv, os, sys, math
import numpy as np
import scipy as sc
# import plotting modules
import matplotlib.pylab as plt 
import matplotlib.patches as patches
import seaborn as sns
from seaborn import *

# function to read the csv file and extract data
def getCSVData(CSVfile):
    data=[]
    with open(CSVfile,newline='') as csvFile:
        reader=csv.reader(csvFile,delimiter=',',quotechar='|',\
                            quoting=csv.QUOTE_NONNUMERIC)
        nrows=0
        for row in reader:
            rawData=row[:]
            csvData=list(filter(None,rawData))
            data.append(csvData)
            nrows = nrows+1
    return data

def EDF(x, sample):
    """Sums over a single sample to calculate its EDF."""
    N = len(sample)
    return np.int32(sample < x).sum()/N

def calc_EDF_values(sample, nPts=100):
    """Calculates the domain and extracts values from the EDF."""
    dom = np.linspace(sample.min(), sample.max(),nPts)
    edf_values = np.array([EDF(dom[n], sample) for n in range(nPts)])
    return dom, edf_values

def calc_EDF_values_dom(sample, dom ):
    #dom = np.linspace(sample.min(), sample.max(),nPts)
    nPts = len(dom)
    edf_values = np.array([EDF(dom[n], sample) for n in range(nPts)])
    return edf_values


def randomSampleFromEDF(dom, EDFvalues, nPts = 100, printMessages=0):
    """Take a randomized sample of a given number of points from the EDF.
    Used for statistical sensitivity testing purposes only."""
    np.random.seed(None)
    sam = np.random.uniform(0,1, nPts)
    ss = np.sort(sam)
    if printMessages==1: print('Random numbers (alphas):',sam)
    if printMessages==1: print('EDF values:',EDFvalues)
    quants = np.zeros(nPts)
    for m in range(nPts):
        idx= np.sum(np.int32(EDFvalues < ss[m]))-1
        quants[m] = dom[idx]
        if printMessages==1: print('index = %d, alpha=%g'%(idx, ss[m]))
    if printMessages==1: print(quants)
    return quants

def Quantile(samp,alpha,nPts=100):
    """Calculates the quantile at b in [0,1] from a distribution function.
    Method added for comparisons and supplementary explanations."""
    dom = np.linspace(samp.min(), samp.max(), nPts)
    EDFvalues = np.array([EDF(dom[n], samp) for n in range(nPts)])
    indx = np.sum(np.int32(EDFvalues < alpha))
    quant = dom[indx]
    #print('index = %d, alpha=%g, quant=%g'%(indx, alpha, quant))
    return dom[indx]

def Quantiles(samp, alphas, nPts=100):
    """Calculates the quantiles from a sample"""
    dom = np.linspace(samp.min(), samp.max(), nPts)
    nAlphas=len(alphas)
    EDFvalues = np.array([EDF(dom[n], samp) for n in range(nPts)])
    quants = np.zeros(nAlphas)
    for m in range(nAlphas):
        indx = np.minimum(nPts-1,np.sum(np.int32(EDFvalues < alphas[m])))
        quants[m] = dom[indx]
        #print('index = %d, alpha=%g, quant=%g'%(indx, alphas[m],quants[m]))
    return quants


def backwardDiffQuotient(x,y):
    """Quotient of backward differences to approximate dy/dx"""
    Nx = len(x); Ny=len(y);
    if Nx==Ny:
        dydx = np.zeros(Ny)
        dydx[1:] = (y[1:]- y[:-1])/(x[1:]- x[:-1])
    else: print('Cannot calculate the difference quotients: x has %d elements and y has %d elements'%(Nx,Ny))
    return dydx

def forwardDiffQuotient(x,y):
    """Quotient of forward differences to approximate dy/dx"""
    N = len(x); Ny=len(y)
    if N==len(y):
        dydx = np.zeros(Ny)
        dydx[1:] = (y[:-1]- y[1:])/(x[:-1]- x[1:])
    else: print('Cannot calculate the difference quotients: x has %d elements and y has %d elements'%(Nx,Ny))
    return dydx

def avgBkwFwdDiffQuotient(x,y):
    dBck = backwardDiffQuotient(x,y)
    dFwd = forwardDiffQuotient(x,y)
    dAvg = (dBck+dFwd)/2
    return dAvg

def centralDiffQuotient(x,y):
    """Quotient of central differences to approximate dy/dx"""
    Nx = len(x); Ny = len(y)
    if (Nx>2)&(Nx==Ny):
        dydx = np.zeros(Ny)
        dydx[1:-1] = (y[2:]- y[:-2])/(x[2:]- x[:-2])
        dydx[0] = (y[1]- y[0])/(x[1]- x[0])
        dydx[-1] = (y[-1]- y[-2])/(x[-1]- x[-2])
    else: print('Cannot calculate the difference quotients: x has %d elements and y has %d elements'%(Nx,Ny))
    return dydx

# -------------------------------------------
# set up class for a single sample (subject)
# -------------------------------------------
class Sample():
    def __init__(self, samplePoints):
        """Get data points from a single sample. Then, sum over the sample to get the EDF."""
        self.sample_pts = np.array(samplePoints) 
        self.sample_nPts = len(samplePoints) 
        self.EDF = lambda u: EDF(x=u, sample= samplePoints)
        self.sample_Min = self.sample_pts.min() 
        self.sample_Max = self.sample_pts.max()
        self.sample_Mean = self.sample_pts.mean() 
        self.sample_centiles = np.linspace(self.sample_Min, self.sample_Max, 100)
        return 

    def calc_EDF(self,nPts=100):
        """Extract values from EDF using specified number of uniformly distributed points."""
        self.ss = np.linspace(self.sample_Min, self.sample_Max, nPts)
        #self.EDF_values = np.array([self.EDF_function(self.ss[m]) for m in range(nPts)]) 
        self.EDF_values = calc_EDF_values_dom(self.sample_pts, self.ss) 
        return self.EDF_values

    def EDF_graph(self,ax,nPts=100,transp=0.6,grpColor='b',lw=2):
        """Calculate and then graph EDF for a single sample (subject)"""
        self.calc_EDF(nPts)
        ax.plot(self.ss,self.EDF_values,':',color=grpColor,lw=lw,alpha=transp)
        return ax
    
    def calc_EMF(self,nPts=100):
        "Empirical mass function from EDF"
        self.EMF_values = centralDiffQuotient(self.calc_EDF(nPts))
        return self.EMF_values

# --------------------------------------------------------------
# set up class to analyze multiple samples (subjects)
# --------------------------------------------------------------
class RepeatedMeasures():
    def __init__(self, sampleList):
        """Get list of multiple subjects in group, and calculate min, max, etc."""
        #print('Sample list \n', sampleList)
        self.nQuants = 100
        self.nSamples = len(sampleList)
        self.samples =[Sample(sampleList[n]) for n in range(self.nSamples) ]
        self.pooled = Sample(np.concatenate( sampleList ) )
        self.domain = np.linspace(self.pooled.sample_Min, self.pooled.sample_Max, self.nQuants)
       #print(self.pooled.sample_pts, '%d total elements in the pooled sample'%self.pooled.sample_nPts)
        #self.poolSamples()
        self.label=''
        return
            
    def calc_meanEDF(self, nPts=100):
        """Calculate EDF for each subject and then average over these to get group EDF"""
        edfvs = np.zeros(nPts)
        for n in range(self.nSamples):
            edfvs += calc_EDF_values_dom(self.samples[n].sample_pts, self.domain)
        
        self.mean_EDF_values = edfvs/self.nSamples
        #print(self.mean_EDF_values)
        return self.mean_EDF_values
    
    def calc_meanEMF(self, nPts=100):
        """Calculate the mean empirical mass function (difference function from the EDF)"""
        self.calc_meanEDF(nPts)
        self.mean_EMF_values= centralDiffQuotient(self.domain, self.mean_EDF_values)
        return self.mean_EMF_values

    def graph_meanEDF(self,ax,nPts=100,sampleColor='black', meanColor='blue',graph=1, graphMass=0):
        """Calculate the mean from the group of sample EDFs and possibly graph it with the sample EDFs"""
        #self.calc_meanEDF(nPts) 
        self.calc_meanEMF(nPts)
        if graph==1:
            ax.plot(self.domain, self.mean_EDF_values, color=meanColor, lw=6, alpha=1.0, label = 'mean EDF '+self.label)
            if graphMass>0:
                ax.plot(self.domain, self.mean_EMF_values, '-', color=meanColor, lw=2, alpha = 0.4, label = 'mean EMF '+self.label)
            ax.legend()
        # Plot the EMFs for each of the samples
        for m in range(self.nSamples):
            s = self.samples[m].sample_pts
            edfMV= calc_EDF_values_dom(s,self.domain)
            if graph==1:
                ax.plot(self.domain, edfMV, ':',color=sampleColor, lw=1, alpha=1.0)
        return self.mean_EDF_values,self.mean_EMF_values
        
    def calc_EDF_pooled(self,nPts=100):
        """Calculates the EDF from the pooled samples. 
        Method added for comparisons and supplementary explanations."""
        self.pooled.calc_EDF(nPts)
        return self.pooled.EDF_values

# Define functions for statistical and sensitivity testing
def meanEDF_WilcoxonRankSumTest(G1,G2, nPts = 25, showTest=0):
    """
    Takes two groups G1 and G2 of samples as inputs (2 lists of containig samples).
    Then it calculates, the mean EDF from each group. 
    Generates a sample of size nPts from the EDFs of the two groups and 
    performs a Wilcoxon Rank Sum test for the pair of samples
    """
    pVals = np.linspace(0,1,nPts)
    Ls = [G1,G2]
    RMGs = list()
    domains = list()
    mEDFs = list()
    rSamples = list()
    for g in Ls:
        rmg = RepeatedMeasures(g)
        rmg.calc_meanEDF()
        doma = rmg.domain 
        mEDF= rmg.mean_EDF_values
        rSamples.append(randomSampleFromEDF(doma, mEDF, nPts))
    #
    print(rSamples)
    s1,s2 = rSamples
    rsTest=sc.stats.ranksums(s1, s2)
    if showTest==1: print(rsTest)
    return rsTest

def meanEDF_WRSTests(G1,G2, nPts = 50, repeats=25, printMessages=0):
    """
    meanEDFWRSTests generates <repeats> pairs of samples, each of size <nPts> from the mean EDF. 
    The sample pairs are used to perform Wilcoxon Rank Sum tests and store the results.
    First, it calculates the mean EDF from each group. 
    Generates a sample of size nPts from the EDFs of the two groups and 
    performs a Wilcoxon Rank Sum test for the pair of samples. 
    Performs the sampling and testing <repeats> times
    """
    ds = list(); mEDFs = list(); Gs =[G1,G2]
    #print(Gs)
    for g in Gs:
        rmg = RepeatedMeasures(g)
        rmg.calc_meanEDF()
        ds.append(rmg.domain); 
        mEDFs.append(rmg.mean_EDF_values)
    #
    d1,d2 = ds; mEDF1, mEDF2 = mEDFs
    if printMessages>0:
        print('Domains:', d1,d2)
        print('meanEDFs:', mEDF1,mEDF2)
        print('\n\n')
    pVals = list(); sts = list()
    for r in range(repeats):
        #print('Generating samples for repeat %d'%r)
        s1 = randomSampleFromEDF(d1,mEDF1,nPts)
        s2 = randomSampleFromEDF(d2,mEDF2,nPts)
        if printMessages>0: print('Sample %d:'%r, s1,'\n',s2,'\n')
        stat, pVal = sc.stats.ranksums(s1, s2)
        pVals.append(pVal); sts.append(stat)
    return np.array(pVals), np.array(sts)

def meanEDF_WRST_sensitivity(G1,G2, Ns = np.arange(5,40,5),repeats=25, plotResults=0, printMessages=0):
    """Runs the Wilcoxon Rank Sum test for the different samples of 
    varying sizes generated from the EDFs."""

    WRS_pval = list()
    for m in range(len(Ns)):
        if printMessages>0: print('Generating amples of size=%d ...'%Ns[m])
        wrst= meanEDF_WRSTests(G1,G2, nPts = Ns[m], repeats=repeats)
        WRS_pval.append(wrst[0])
        
    WRS_pval = np.array(WRS_pval)
    d1, d2 = np.shape(WRS_pval)
    if printMessages>0: print('pValues in a shape of %d by %d'%(d1,d2))
    WRS_up = WRS_pval.max(axis=1)
    WRS_lo = WRS_pval.min(axis=1)
    WRS_mean = WRS_pval.mean(axis=1)
    #    
    if plotResults>0:
        plt.figure(figsize=(12,5)); plt.ioff(); 
        for m in range(len(Ns)):
            theseP = WRS_pval[m,:]
            ns = Ns[m]*np.ones(repeats)
            plt.plot(ns,theseP, '.',ms=8)
        plt.plot([Ns.min(),Ns.max()],[0.05,0.05],'k',linestyle='--',linewidth=2,alpha=0.35,label=r'$\alpha$=0.05')
        #plt.plot(Ns, WRS_up,'k-',alpha=0.4,lw=1)
        #plt.plot(Ns, WRS_lo,'k-',alpha=0.4,lw=1)
        plt.plot(Ns, WRS_mean,'k-',alpha=0.4,lw=3,label='mean')
        plt.ylabel('p-value')
        plt.xlabel('Theoretical sample size')
        plt.xlim(4,41)
        plt.legend()
        plt.ion(); plt.draw(); plt.show()
    return WRS_pval


import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.signal import stft
from scipy import sparse
from scipy.sparse.linalg import spsolve

"""
We use a kaiser window to balance the main-lobe width and side lobe level. 
see here, ~https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html#numpy-kaiser~
as well, ~https://en.wikipedia.org/wiki/Window_function~
"""
window_np = lambda tnp,nw: np.array([tnp[...,0],np.kaiser(len(tnp),nw)*tnp[...,1]]).T

def  bg_als_opt(y,lam,p,dorder):
    """
    Baseline correction for removing 'untrue' low-frequency components due to the signal instability. 
    Method:'Paul H. C. Eilers, Parametric Time Warping  Anal. Chem. 2004, 76, 2, 404 - 411 ' ~pubs.acs.org/doi/10.1021/ac034800e~
    codes are from ~https://stackoverflow.com/questions/29156532/python-baseline-correction-library~ 
    python should have a package of symmetric baseline correction by least-squares as well 

    Given a column of data y,
    it needs three parameters to correct the baseline
    lam: \lambda, smoothness, 10**x. works with exponential order. 
    p: possibility. 0-1.0.  0.5 is symmetric for top and bottom sides. 0 is at the bottom, 1 is at the top of the spectrum
    dorder: control polyorder of the differential order.
    The background curve is returned by the function. 
    """
    niter = 10   # iteration number to solve the least-square equation.
    L = len(y)
    D = sparse.csr_matrix(np.diff(np.eye(L),n=dorder))     # build up a compressed sparse matrix of difference polynomials  
    D = lam * D.dot(D.transpose())   # add smoothness and calculate the D^2
    w = np.ones(L)  # one matrix of the y intensity 
    W = sparse.spdiags(w, 0, L, L)   # build up a sparse matrix to for off diangnol part
    for i in range(niter):
        W.setdiag(w)
        Z = W + D  # sum up diagnol and offdiagnol parts.
        z = spsolve(Z, w*y)   # solve the least-square mean
        w = p * (y > z) + (1-p) * (y < z)  # give the weight of the up and down parts, and go into the second iteration.
    return z

def fft_tau2freq(tnp,Zpad): 
    """
    This is the main function for fft section. 
    We do fft of a windowed experimetnal oscillation spectrum. 
    A number for the power of 2 is need for padding zero.
    The frequency spectrum in numpy array is the return format. 
    """
    ln=len(tnp)
    N=2**int(Zpad)
    if ln > N:
        N=2**int(np.log2(ln)+1)
    """
    First, we solve a N number for zero padding. We aslo keep an even number of input points for FFT.  
    Notes that the symmetry is highest when n is a power of 2, and the transform is therefore most efficient for these sizes.
    ~https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft~
    """
    ffty = np.fft.fft(tnp[...,1],n=N,)[:int(N/2)]
    freqy = np.abs(ffty)*2/ln  # normalize the amplitude by 1/Ln
    Fs = 10**15/(np.abs(tnp[-1,0]-tnp[0,0])/ln)/N
    freqx = np.arange(int(N)/2)*Fs/(constants.c*100) # thes two lines are for converint to cm-1   
    fftnp = np.array([freqx,freqy]).T
    return (fftnp)


def osc_sftp(tnp,wtime):
    """
    This is the function for short time ft map using scipy
    a oscillation and window time parameter are needed.
    """
    tstep=np.mean(np.abs(np.diff(tnp[...,0])))   # got the temporal intervel
    window_n = wtime//tstep+2  # solve data point needs for a window intervel

    """
    ~https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html~
    padding zero 2**3 times than the signal. apply a kaiser window.  overlap most of the data points. 
    """
    f,t,Zxx = stft(tnp[...,1], fs=10**15/tstep, noverlap=window_n-1,nperseg=window_n, nfft=2**int(np.log2(window_n)+1+3), window=('kaiser',4.5)) 
    return(f/(constants.c*100), t*10**15,np.transpose(np.abs(Zxx)))  # conver units
 
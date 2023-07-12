import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.signal import stft

"""
We use a kaiser window to balance the main-lobe width and side lobe level. 
see here, ~https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html#numpy-kaiser~
as well, ~https://en.wikipedia.org/wiki/Window_function~
"""
window_np = lambda tnp,nw: np.array([tnp[...,0],np.kaiser(len(tnp),nw)*tnp[...,1]]).T

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
 
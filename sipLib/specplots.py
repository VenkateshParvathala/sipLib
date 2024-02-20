'''
		Author: 				Venkatesh Parvathala
		Affiliated to: 	SipLab - IIT Hyderabad
		Contact:				ee22resch01005@iith.ac.in
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotSpectrogram(wavs, n_rows=None, n_cols=None, n_fft=2048, win_length=1024, hop_length=256, fs=16000, Vmin=None, Vmax=None, n_sd=None, origin='lower', aspect='auto', colorbar=False):
    '''
    Plots the spectrograms by computing some parameters automatically
    
    Arguments:
        specs: list - All the spectrograms in a list
        n_rows: int - Number of rows in the plot
        n_cols: int - Number of cols in the plot
        vmin: int - Minimum value of the spectrogram
        vmax: int - Maximum value of the spectrogram
        origin: str - Where to start the (0,0) index
        aspect: str - Aspect ratio of the plot
        colorbar: bool - whether to keep colorbar or not
    '''
    specs = []
    for wav in wavs:
        spec = np.log(np.abs(librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length))+1e-5)
        specs.append(spec)
    n_plots = len(specs)
    
    vmin = Vmin
    vmax = Vmax
    if n_rows is None: n_rows=len(specs)
    if n_cols is None: n_cols=1
    if n_rows*n_cols != n_plots:
        print("Error: The number of spectrograms is expected to be rows*cols({:d}) but got {:d}".format(n_rows*n_cols, n_plots))
        exit(0)
    specs = [specs[n*n_cols:(n+1)*n_cols] for n in range(n_rows)]
    fig, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            spec = specs[i][j]
            yu = spec.shape[0]
            xu = spec.shape[1]
            if n_sd is None:
                if Vmin is None: vmin=None#np.mean(spec.flatten())-n_sd*np.std(spec.flatten())
                if Vmax is None: vmax=None#np.mean(spec.flatten())+n_sd*np.std(spec.flatten())
            else:
                if Vmin is None: vmin=np.mean(spec.flatten())-n_sd*np.std(spec.flatten())
                if Vmax is None: vmax=np.mean(spec.flatten())+n_sd*np.std(spec.flatten())                
            if n_rows==1 and n_cols==1:
                pos = ax.imshow(spec, vmin=vmin, vmax=vmax, origin=origin, aspect=aspect)
                if colorbar: fig.colorbar(pos, ax=ax, orientation='vertical');
                plt.setp(ax, yticks=np.linspace(0, yu, 5), yticklabels=np.linspace(0, fs//2000, 5).astype('i'))
                plt.setp(ax, xticks=np.linspace(0, xu, 5), xticklabels= np.round((np.linspace(0, xu, 5)*hop_length)/fs, 1))
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Frequency (kHz)')
            elif n_rows==1:
                pos = ax[j].imshow(spec, vmin=vmin, vmax=vmax, origin=origin, aspect=aspect)
                if colorbar: fig.colorbar(pos, ax=ax[j], orientation='vertical');
                if j==0:
                    plt.setp(ax[j], yticks=np.linspace(0, yu, 5), yticklabels=np.linspace(0, fs//2000, 5).astype('i'))
                    ax[j].set_ylabel('Frequency (kHz)')
                else:
                    plt.setp(ax[j], yticks=[], yticklabels=[])
                plt.setp(ax[j], xticks=np.linspace(0, xu, 5), xticklabels= np.round((np.linspace(0, xu, 5)*hop_length)/fs, 1))
                ax[j].set_xlabel('Time (sec)')
                    
            elif n_cols==1:
                pos = ax[i].imshow(spec, vmin=vmin, vmax=vmax, origin=origin, aspect=aspect)
                if colorbar: fig.colorbar(pos, ax=ax[i], orientation='vertical');
                plt.setp(ax[i], yticks=np.linspace(0, yu, 5), yticklabels=np.linspace(0, fs//2000, 5).astype('i'))
                ax[i].set_ylabel('Frequency (kHz)')
                if i==n_rows-1:
                    plt.setp(ax[i], xticks=np.linspace(0, xu, 5), xticklabels= np.round((np.linspace(0, xu, 5)*hop_length)/fs, 1))
                    ax[i].set_xlabel('Time (sec)')
                else:
                    plt.setp(ax[i], xticks=[], xticklabels=[])
                
            else:
                pos = ax[i,j].imshow(spec, vmin=vmin, vmax=vmax, origin=origin, aspect=aspect)
                if colorbar: fig.colorbar(pos, ax=ax[i,j], orientation='vertical');
                if j==0:
                    plt.setp(ax[i,j], yticks=np.linspace(0, yu, 5), yticklabels=np.linspace(0, fs//2000, 5).astype('i'))
                    ax[i,j].set_ylabel('Frequency (kHz)')
                else:
                    plt.setp(ax[i,j], yticks=[], yticklabels=[])
                if i==n_rows-1:
                    plt.setp(ax[i,j], xticks=np.linspace(0, xu, 5), xticklabels= np.round((np.linspace(0, xu, 5)*hop_length)/fs, 1))
                    ax[i,j].set_xlabel('Time (sec)')
                else:
                    plt.setp(ax[i,j], xticks=[], xticklabels=[])
                
    plt.show()

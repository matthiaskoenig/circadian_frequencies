"""
Helper functions for circadian analysis.
"""
from __future__ import absolute_import, print_function, division
from six import iteritems
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import scipy.fftpack

def tf_data(z0_mu=1, z0_cv=0.1, phi_mu=0, phi_cv=0, T_mu=24, T_cv=0.01, sampling=1, duration=48, n_samples=100):
    """ Create data for single transcription factor.
    
    :param z0_mu: amplitude
    :param z0_cv: coefficient of variation amplitude
    :param phi_mu: phase in degree
    :param phi_cv: coefficient of variation phase degree 
    :param T_mu: circadian period in [h]
    :param T_cv: coefficient of variation circadian period in [h]
    :param sampling: sampling interval [h]
    :param duration: duration of timecourse
    :param n_samples: number of samples from distributions
    :return: 
    """
    """ 

    param
    """
    mu, sigma = 0, 0.1  # mean and standard deviation
    z0_sigma = z0_cv * z0_mu
    z0 = np.random.normal(z0_mu, z0_sigma, n_samples)
    T_sigma = T_cv * T_mu
    T = np.random.normal(T_mu, T_sigma, n_samples)
    phi_sigma = phi_cv * phi_mu
    phi = np.random.normal(phi_mu, phi_sigma, n_samples) * 2 * np.pi / T_mu

    # every column is one timepoint, every row one sample
    n_points = np.int(np.round(1.0 * duration / sampling))
    data = np.zeros(shape=(n_samples, n_points))
    for k in range(n_samples):
        row = z0_mu + z0[k] * np.sin(2 * np.pi / T[k] * np.linspace(0, duration, n_points) + phi[k])
        data[k, :] = row

    return data


def fft_data(data):
    """ FFT of data to find frequencies.
    
    :param x: 
    :param data: 
    :return: 
    """

    n_samples, n_points = data.shape
    yf = np.zeros(shape=(n_samples, n_points))

    for k in range(n_samples):
        # fft of row
        row = np.array(data[k, :])
        yf[k, :] = scipy.fftpack.fft(row)
    return yf


if __name__ == "__main__":
    duration = 48
    sampling = 1
    n_points = np.int(np.round(1.0 * duration / sampling))
    n_samples = 100

    d1 = tf_data(sampling=1, duration=duration, n_samples=n_samples)
    d2 = tf_data(sampling=1, duration=duration, phi_mu=180, n_samples=n_samples)
    dsum = d1+d2
    datasets = [d1, d2, dsum]

    # x_fft = np.linspace(0.0, 1.0 / (2.0 * duration), int(np.round(n_points /2.0)))

    # ffts = [fft_data(data) for data in datasets]
    ffts = [fft_data(data) for data in [d1]]


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    for d_index, data in enumerate(datasets):
        if d_index == 0:
            dcolor = 'blue'
        elif d_index == 1:
            dcolor = 'darkgreen'
        elif d_index == 2:
            dcolor = 'black'
        print(d_index, dcolor)
        for k in range(n_samples):
            ax1.plot(data[k, :], color=dcolor, alpha=0.5)

    for d_index, data in enumerate(ffts):
        if d_index == 0:
            dcolor = 'blue'
        elif d_index == 1:
            dcolor = 'darkgreen'
        elif d_index == 2:
            dcolor = 'black'
        print(d_index, dcolor)
        for k in range(n_samples):
            # todo axis in frequencies
            ax2.plot(data[k, :], color=dcolor, alpha=0.5)

    ax1.set_ylabel('amplitude [a.u.]')
    ax1.set_ylabel('time [h]')

    plt.show()

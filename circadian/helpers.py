"""
Helper functions for circadian analysis.
"""
from __future__ import absolute_import, print_function, division
from six import iteritems
import numpy as np
import scipy as sp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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
        row = 1.2*z0_mu + z0[k] * np.sin(2 * np.pi / T[k] * np.linspace(0, duration, n_points) + phi[k])
        data[k, :] = row

    return data


def fft_data(data, sampling):
    """ FFT of data to find frequencies.
    
    :param x: 
    :param data: 
    :return: 
    """

    n_samples, n_points = data.shape
    fft = np.zeros(shape=(n_samples, n_points))
    yf = np.zeros(shape=(n_samples, int(np.round(n_points/2))) )

    for k in range(n_samples):
        # fft of row
        row = np.array(data[k, :])
        fft = scipy.fftpack.fft(row-np.mean(row))

        # The returned complex array contains ``y(0), y(1), ..., y(n - 1)`` where
        # y(j) = (x * exp(-2 * pi * sqrt(-1) * j * np.arange(n) / n)).sum()

        yf[k, :] = 2.0 / n_points * np.abs(fft[:n_points // 2])

    return yf

def full_dataset():
    duration = 96
    sampling = 1
    n_samples = 100
    n_points = np.int(np.round(1.0 * duration / sampling))


    d1 = tf_data(sampling=1, duration=duration, n_samples=n_samples)
    n_phis = 27
    xf = np.linspace(0.0, 1.0 / (2.0 * sampling), n_points // 2)
    phis = np.linspace(0, 360, n_phis)

    spectrum = np.zeros(shape=(len(phis), n_points//2))

    for k, phi in enumerate(phis):
        d2 = tf_data(sampling=1, duration=duration, phi_mu=phi, n_samples=n_samples)
        dsum = d1 + d2
        yf = fft_data(dsum, sampling=sampling)
        spectrum[k, :] =  np.mean(yf, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for k, phi in enumerate(phis):
        ax1.plot(xf, spectrum[k,:], alpha=0.5)

    plt.show()


if __name__ == "__main__":

    # full_dataset()
    duration = 96
    sampling = 1
    phi = 180
    z0 = 1.0
    T = 12
    n_samples = 100
    n_points = np.int(np.round(1.0 * duration / sampling))

    d1 = tf_data(sampling=1, duration=duration, n_samples=n_samples)
    d2 = tf_data(sampling=1, duration=duration, z0_mu=z0, phi_mu=phi, T_mu=T, n_samples=n_samples)
    dsum = d1*d2
    datasets = [d1, d2, dsum]

    # x_fft = np.linspace(0.0, 1.0 / (2.0 * duration), int(np.round(n_points /2.0)))

    # ffts = [fft_data(data) for data in datasets]
    xf = np.linspace(0.0, 1.0 / (2.0 * sampling), n_points//2)
    yfs = [fft_data(data, sampling=sampling) for data in [dsum]]

    alpha = 0.3
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for d_index, data in enumerate(datasets):
        if d_index == 0:
            dcolor = 'blue'
        elif d_index == 1:
            dcolor = 'darkgreen'
        elif d_index == 2:
            dcolor = 'black'
        print(d_index, dcolor)
        for k in range(n_samples):
            ax1.plot(data[k, :], color=dcolor, alpha=alpha)

    for d_index, data in enumerate(yfs):
        dcolor='gray'

        for k in range(n_samples):
            # todo axis in frequencies
            ax2.plot(xf, data[k, :], '-o', color=dcolor, alpha=alpha)
            ax3.plot(1/xf, data[k, :], '-o', color=dcolor, alpha=alpha)
            # ax2.plot(1/xf, data[k, :], '-o', color=dcolor, alpha=alpha)
        ax2.plot(xf, np.mean(data, axis=0), '-o', color='black', linewidth=2)
        ax3.plot(1/xf, np.mean(data, axis=0), '-o', color='black', linewidth=2)

        for xline in [8, 12, 24]:
            ax3.plot([xline, xline], [0, 3.0], '--', color="black")

    ax1.set_ylabel('amplitude [a.u.]')
    ax1.set_xlabel('time [h]')
    ax2.set_xlabel('frequency [1/h]')
    ax3.set_xlabel('time [h]')

    plt.show()

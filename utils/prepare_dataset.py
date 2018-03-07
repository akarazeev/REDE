# File name: prepare_dataset.py
# Author: Anton Karazeev <anton.karazeev@gmail.com>
#
# This file is part of REDE project (https://github.com/akarazeev/REDE)
#
# Description: content of this file was used to build a dataset
# callsed REDE - Reverse Engineering in Dispersion Engineering.
# Files from utils/matlab_data/ were used as raw data. These data
# contain information about system's simulations with given set of
# parameters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import tqdm
import pickle
import scipy.io as spio


def preproc(freqs, modes):
    c = 2.99792458e8
    T0 = 282e12
    N = 20000

    modes_total = np.linspace(min(modes), max(modes), N)
    tck = interpolate.CubicSpline(modes, freqs)
    omega_total = tck(modes_total)

    h = (max(modes) - min(modes)) / (N - 1)

    D1_total = np.diff(omega_total) / h
    D2_total = np.diff(D1_total) / h

    modes_total = modes_total[:-1]
    omega_total = omega_total[:-1]

    ind = np.argmin(abs(T0 - omega_total))
    w0 = omega_total[ind]
    m0 = modes_total[ind]
    D1 = D1_total[ind]

    omega_grid = w0 + (D1 * (modes_total - m0))

    delta = (omega_total - omega_grid) % D1
    delta_omega_total = delta - (np.sign(delta - (D1 / 2)) * D1 * (abs(delta) > (D1 / 2)))
    lambda_grid = c / omega_total

    return omega_total, delta_omega_total, D1_total, D2_total


# Load data.
mat = spio.loadmat('matlab_data/full_set.mat', squeeze_me=True)

struct = mat['str']
header = ['id']
header.extend(struct[0][1].dtype.names)
header

# Create DataFrame.
dataset = []

for i in range(len(struct)):
    tmp = [int(struct[i][0])]
    tmp.extend([float(struct[i][1][name]) for name in header[1:]])
    dataset.append(tmp)

df_data = pd.DataFrame(data=dataset, columns=header)
df_data.head()

# Generate dataset.
frequencies_modes_list = []
parameters_list = []
images = []

for i in tqdm.tqdm(range(len(struct))):
    # Parameters.
    sample_id = int(struct[i][0])
    parameters = df_data[df_data['id'] == sample_id].values[0][1:]
    parameters_list.append(parameters)

    # Frequencies and modes.
    freqs, modes = struct[i][2][:, 0].real, struct[i][2][:, 2].real
    frequencies_modes_list.append((freqs, modes))

    # Images.
    omega_total, delta_omega_total, D1_total, D2_total = preproc(freqs, modes)
    fig = plt.figure(figsize=(2, 1))
    fig.add_subplot(111)
    plt.xlim((130, 430))
    plt.ylim((-500, 500))
    plt.axis('off')
    img = plt.scatter(omega_total * 1e-12, delta_omega_total * 1e-9, s=0.01)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[5:-5, 20:-13, 0]
    data = (data < 200) * 255
    data = data.astype('uint8')
    images.append(data)
    plt.close()

# Convert to np.array.
images = np.array(images)
frequencies_modes_list = np.array(frequencies_modes_list)
parameters_list = np.array(parameters_list)

with open('dataset/1056-5-parameters.pkl', 'wb') as f:
    pickle.dump(parameters_list, f)

with open('dataset/frequencies_modes.pkl', 'wb') as f:
    pickle.dump(frequencies_modes_list, f)

with open('dataset/1056-62-111-images.pkl', 'wb') as f:
    pickle.dump(images, f)

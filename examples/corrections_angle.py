import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging
import os, sys
import argparse
import multiprocessing as mp
import models.HDF5Dataset as H
import torch

def correct2D(dpath, BATCH_SIZE):

    ## Load and make them iterable
    data = H.HDF5Dataset(dpath, 'ecal')
    energies = data['energy'][:].reshape(len(data['energy'][:]))
    angles = data['theta'][:].reshape(len(data['theta'][:]))
    layers = data['layers'][:]

    training_dataset = tuple(zip(layers, energies, angles))


    dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=0)
    dataiter = iter(dataloader)

    batch = next(dataiter, None)

    if batch is None:
        dataiter = iter(dataloader)
        batch = dataiter.next()

    real_data = batch[0] # 32x30 calo layers
    real_data = real_data.numpy()

    real_energy = batch[1]
    real_energy = real_energy.unsqueeze(-1)
    real_energy = real_energy.numpy()

    real_angle = batch[2]
    real_angle = real_angle.unsqueeze(-1)
    real_angle = real_angle.numpy()


    real_d = real_data.mean(axis=0)
    for layer_pos in range(len(real_d)):
        for row_pos in range(len(real_d[0])):
            if real_d[layer_pos, row_pos].sum().item() < 0.001:
                for i in range(row_pos, 0, -1):
                    real_data[:, layer_pos, i] = real_data[:, layer_pos, i-1]
    real_data = real_data[:, :, 1:-1]
    

    return real_data, real_energy, real_angle

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input file name')
parser.add_argument('--output', type=str, required=True, help='output name of preprocessed file')
parser.add_argument('--batchsize', type=int, required=True, help='batchsize')

opt = parser.parse_args()
inputfile = str(opt.input)
outputfile = str(opt.output)
BS = int(opt.batchsize)

pixs, energy, angle = correct2D(inputfile, BS)

#Open HDF5 file for writing
hf = h5py.File(outputfile, 'w')
grp = hf.create_group("ecal")


## write to hdf5 files
grp.create_dataset('energy', data=energy)
grp.create_dataset('layers', data=pixs)
grp.create_dataset('theta', data=angle)



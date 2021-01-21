from pyLCIO import EVENT, UTIL, IOIMPL, IMPL
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
import h5py 
import awkward1 as ak 
import argparse
import math

from functions import CellIDDecoder

## these are the layer positions in y[mm] (depth of the calorimeter)
hmap = np.array([1811, 1814, 1824, 1827, 1836, 1839, 1849,
                    1852, 1861, 1864, 1873, 1877, 1886, 1889, 1898, 1902,
                    1911, 1914, 1923, 1926, 1938, 1943, 1955, 1960,
                    1971, 1976, 1988, 1993, 2005, 2010])


def fill_record(inpLCIO, collection):
    """this function reads all events in LCIO file and put them into awkward array"""

    ## open LCIO file
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open( inpLCIO )

    ## create awkward array
    b = ak.ArrayBuilder()

    ## start looping
    nEvt = 0
    for evt in reader:
        nEvt += 1
        if nEvt > nevents:
            break

        
        ## First thing: MC particle collection
        b.begin_list()
        mcparticle = evt.getCollection("MCParticle")    
        
        ## fill energy for each MCParticle (in this case just an incoming photon)
        for enr in mcparticle:
            b.begin_record()
            b.field("E")
            b.real(enr.getEnergy())
            ## calculate polar angle theta and fill
            b.field("theta")
            pVec = enr.getMomentum()
            theta = math.pi/2.00 - math.atan(pVec[2]/pVec[1])
            b.real(theta)
            b.end_record() 
    
        ## ECAL barrel collection
        ecalBarrel = evt.getCollection(collection)
        cellIDString = ecalBarrel.getParameters().getStringVal("CellIDEncoding")
        decoder = CellIDDecoder( cellIDString ) 
        ##

        for hit in ecalBarrel:

            l = decoder.layer( hit.getCellID0() ) ## get the layer information from CellID0 
            e = hit.getEnergy() 
            pos = hit.getPosition()

            ## start filling a record with all relevant information
            b.begin_record() 
            b.field("x")
            b.real(pos[0])
            b.field("y")
            b.real(pos[1])
            b.field("z")
            b.real(pos[2])
            b.field("e")
            b.real(e * 1000)
            b.field("layer")
            b.integer(l)
            b.end_record() 

        b.end_list()

    ### Example:
    # Get the incident energy of the first event --> b[0].E
    # Get the x positions of the the first event --> b[0].x 

    return b


def fill_numpy(record):
    """this function reads the awkward array and edits for our needs: Projecting into 32x30 grid"""
    
    #defined binning
    binX = np.arange(-81, 82, 5.088333)
    #binZ = np.arange(-77, 78, 5.088333)
    #new z-shifted bin in Z
    #binZ = np.arange(-70, 83, 5.088333)
    #new z- asymmetric bin
    #binZ = np.arange(-90, 65, 5.088333)
    binZ = np.arange(-80, 75, 5.088333)
    #binZ = np.arange(-45, 110, 5.088333)
    #binZ = np.arange(-35, 121, 5.088333)

    ## Unable to escape using python list here. But we can live with that.
    l = []
    E = []
    theta = []
    for i in range(0, nevents):

        #Get hits and convert them into numpy array 
        z = ak.to_numpy(record[i].z)
        x = ak.to_numpy(record[i].x)
        y = ak.to_numpy(record[i].y)
        e = ak.to_numpy(record[i].e)

        #Get indicent photon energies
        incE = ak.to_numpy(record[i].E)
        E.append(incE.compressed()[0])

        #Get polar angle theta of the gun
        inTh = ak.to_numpy(record[i].theta)
        theta.append(inTh.compressed()[0])



        layers = []
        #loop over layers and project them into 2d grid.
        for j in range(0,30):
            idx = np.where((y <= (hmap[j] + 0.9999)) & (y > (hmap[j] + 0.0001)))
            xlayer = x.take(idx)[0]
            zlayer = z.take(idx)[0]
            elayer = e.take(idx)[0]
            H, xedges, yedges = np.histogram2d(xlayer, zlayer, bins=(binX, binZ), weights=elayer)
            layers.append(H)

   
        l.append(layers)
    
    ## convert them into numpy array
    shower = np.asarray(l)
    e0 = np.reshape(np.asarray(E), (-1,1))
    t0 = np.reshape(np.asarray(theta), (-1,1))

    return shower, e0, t0

if __name__=="__main__":

    ## get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input LCIO file')
    parser.add_argument('--maxEvent', type=int, default=10, help='Number of events read')
    parser.add_argument('--collection', type=str, required=True, help='input collection in the LCIO file')
    parser.add_argument('--output', type=str, required=True, help='output name hdf5 file')

    opt = parser.parse_args()

    inpLCIO = str(opt.input)
    nevents = int(opt.maxEvent)
    collection = str(opt.collection)
    out = str(opt.output)

    # STEP 1: create awkward array from LCIO
    record = fill_record(inpLCIO, collection)
    # STEP 2: convert them into numpy arrays
    showers, e0, t0 = fill_numpy(record)
    
    #Open HDF5 file for writing
    hf = h5py.File(out, 'w')
    grp = hf.create_group("ecal")


    ## write to hdf5 files
    grp.create_dataset('energy', data=e0)
    grp.create_dataset('theta', data=t0)
    grp.create_dataset('layers', data=showers)

    #close file
    hf.close()

    

    

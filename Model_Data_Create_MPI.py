#!/usr/bin/env python
import os,pdb,h5py
from mpi4py import MPI
import numpy as np
from optparse import OptionParser
import scipy.io as sio


"""
Author : Mahesh Balasubramanian (Adapted from Alex Bujan (adapted from Kris Bouchard)) 
Date    : 05/16/2017
"""

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("--mdlW",type="string",default="ExpI",\
        help="weight distribution (options: 'Gaus','ExpI','Clst1','Clst2','Lap','Uni','ID')")
    parser.add_option("--mdlsz",type="int",default=100,\
        help="number of non-null dimensions in the model")
    parser.add_option("--v1",type="float",default=.0,\
        help="magnitude of the noise in the model")
    parser.add_option("--v2",type="float",default=4.,\
        help="ratio of null/non-null dimensions in the model")
    parser.add_option("--v3",type="int",default=3,\
        help="ratio of samples/parameters in the model")
    parser.add_option("--store",action="store_true",dest="store",\
        help="store results to file")
    parser.add_option("--saveAs",type="string",default='hdf5',\
        help="File format to store the data. Options: hdf5(default), mat, txt")
    parser.add_option("--path",type="string",default=os.getcwd(),\
        help="path to store the results (default: current directory)")
    parser.add_option("--seed",type="int",default=np.random.randint(9999),\
        help="seed for generating pseudo-random numbers (default: random seed)")

    (options, args) = parser.parse_args()

    if options.store:
        store=True
    else:
        store=False

    Model_Data_Create(mdlW=options.mdlW,v1=options.v1,\
                    v2=options.v2,v3=options.v3,\
                    mdlsz=options.mdlsz,store=store,\
                    path=options.path,seed=options.seed,\
                    saveAs=options.saveAs)

def Model_Data_Create(mdlW,v1,v2,v3,mdlsz=100,seed=np.random.randint(9999),\
                        store=False,path=os.getcwd(),saveAs='hdf5'):

    """
    Model_Data_Create
    -----------------

    Creates data samples using different underlying coefficient distributions.

    Input:
        -mdlW       : weight distribution; options: 'Gaus','ExpI'(default),
                        'Clst1','Clst2','Lap','Uni','ID'
        -mdlsz      : number of non-null dimensions in the model
        -v1         : magnitude of the noise in the model
        -v2         : ratio of null/non-null dimensions in the model
        -v3         : ratio of samples/parameters in the model
        -store      : store results to file
        -saveAs     : file format to store the data; options: hdf5(default),
                        mat, txt
        -path       : path to store the results (default: current directory)
        -seed       : seed for generating pseudo-random numbers (default: 
                        a different random seed is generated every time

    Output:
        - X         : design matrix
        - y         : response
        - Wact      : true weights
    """

    np.random.seed(seed)

    '''
    Get communicator
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    minSize = nMP*nrnd
    if size!=minSize:
        raise ValueError('The number of processes must be %i'%minSize)
    if n<0 or m<0:
        raise ValueError('Introduce a valid number of model dimensions and data samples')

    if rank==0:
        '''
        #create model weights
        #--------------------
        '''
        if mdlW=='Gaus':
            mn =-4
            mx = 4
            W  = (mx-mn)*np.random.normal(size=(mdlsz,1))
        elif mdlW=='Uni':
            mn =-5
            mx = 5
            W  = mn+(mx-mn)*np.random.uniform(size=(mdlsz,1))
        elif mdlW=='Lap':
            mn = -2
            mx = 2
            W = np.exp(np.linspace(mn,mx,np.floor(mdlsz/2.)))
            W = np.hstack((-1*W,W))[:,np.newaxis]
        elif mdlW=='ExpI':
            mn = -2
            mx = 2
            W  = np.exp(np.linspace(mn,mx,np.floor(mdlsz/2.)))
            W1 = W-np.max(W)-.1*np.max(W)
            W2 = np.abs(W-np.max(W)-.1*np.max(W))
            W  = np.hstack((W1,W2))[:,np.newaxis]
        elif mdlW=='Clst1':
            mn = 5
            mx = 30
            lp = np.linspace(mn,mx,6)
            for i in xrange(5):
                lpW = lp[i]+np.random.uniform(size=(np.floor(mdlsz/5.),1))
                if i==0:
                    W = lpW
                else:
                    W = np.vstack((W,lpW))
        elif mdlW=='Clst2':
            mn = 3
            mx = 20
            lp = np.linspace(mn,mx,6)
            for i in xrange(5):
                lpW = np.exp(np.linspace(lp[i],lp[i+1],np.floor(mdlsz/10.)))[:,np.newaxis]
                if i==0:
                    W = np.vstack((-lpW,lpW))
                else:
                    W = np.vstack((W,-lpW,lpW))
        elif mdlW=='ID':
            W = 10*np.ones(size=(mdlsz,1))

        #total number of data samples
        nd = v3*(mdlsz+mdlsz*v2)

    comm.Barrier()
    comm.Bcast (nd, MPI.INT)
    comm.Bcast (W, MPI.DOUBLE)

    '''
    #generate input data
    #-------------------
    '''
    #data for non-null dimensions
    Dat     = 3*np.random.normal(size=(mdlsz,nd//size))
    #data for null dimensions
    Dat2    = 3*np.random.normal(size=(1+round(v2*mdlsz),nd//size))
    #design matrix // input data // non-null and null dimensions
    DDat    = np.vstack((Dat,Dat2)).T
    
    if rank==0:
        '''
        #ground truth
        #------------
        '''
        #dim(Wact)<-mdlsz+mdlsz*v2+1
        tmp = np.zeros((1+round(v2*mdlsz),1))
        Wact    = np.vstack((W,tmp))
        
    y = np.dot(W.T,Dat)+v1*np.sum(np.abs(W))*np.random.normal(size=nd//size)
    y-=np.mean(y)

    name = '%s_%.1f_%i_%i'%(mdlW,v1,v2,v3)
    fx=h5py.File('%s/Model_Data_%s.h5'%(path,name),'w',driver='mpio',comm=MPI.COMM_WORLD)

    if store:
        if saveAs=='hdf5':
            with fx as f:
                g = f.create_group('data')
                g.create_dataset(name='X',data=DDat,dtype=np.float64,\
                                shape=DDat.shape,compression="gzip")
                g.create_dataset(name='y',data=np.ravel(y),dtype=np.float64,\
                                shape=np.ravel(y).shape,compression="gzip")
                g.create_dataset(name='Wact',data=Wact,dtype=np.float64,\
                                shape=Wact.shape,compression="gzip")
        elif saveAs=='txt':
            np.savetxt('%s/X_%s.txt'%(path,name),DDat.astype(np.float32))
            np.savetxt('%s/y_%s.txt'%(path,name),np.ravel(y).astype(np.float32))
            np.savetxt('%s/Wact_%s.txt'%(path,name),Wact.astype(np.float32))

        elif saveAs=='mat':
            sio.savemat('%s/Model_Data_%s.mat'%(path,name),\
                        {'X'    : DDat.astype(np.float32),\
                         'y'    : np.ravel(y).astype(np.float32),\
                         'Wact' : Wact.astype(np.float32)})

        print '\nData Model:'
        print '\t* No covariates:\t%i'%DDat.shape[0]
        print '\t* No samples   :\t%i'%DDat.shape[1]
        print 'Data stored in %s'%path
    else:
        return DDat,np.ravel(y),Wact

    MPI_Finalize()

if __name__=='__main__':
    main()


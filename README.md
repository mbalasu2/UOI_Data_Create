# UOI_Data_Create

This folder has 3 files.

1. Model_Data_Create.py -- This file is a serial version of synthetic data creation. This breaks when v3 in >=30 because of the insufficient memory in edison. In cori it may break at some point.
    --status=working

2. Model_Data_Create_MPI.py -- This is a parallel version of the serial version. Uses MPI to distribute the data creation calculation and writes in parallel to the output file. Transpose is done by each process before writing and not during writing of output like the serial version.
     --status=to_be_tested
     
3. Model_Data_Create_MPI_v2.py -- This uses collective i/o for storing data from https://github.com/valiantljk/h5py/blob/master/examples/collective_io.py page. This is supposed to be more efficient than the previous version of parallel data creation and more intuitive. Use this version to create data. 
       --status=to_be_tested

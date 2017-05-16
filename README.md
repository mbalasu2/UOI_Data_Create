# UOI_Data_Create

This folder has two files.

Model_Data_Create.py -- This file is a serial version of synthetic data creation. This breaks when v3 in >=30 because of the insufficient memory in edison. In cori it may break at some point.
--status working

Model_Data_Create_MPI.py -- This is a parallel version of the serial version. Uses MPI to distribute the data creation calculation and writes in parallel to the output file. Transpose is done by each process before writing and not during writing of output like the serial version.

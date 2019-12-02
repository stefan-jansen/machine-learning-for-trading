## Efficient data storage with pandas

The notebook [storage_benchmark](storage_benchmark.ipynb) compares the main storage formats for efficiency and performance. 

In particular, it compares:
- CSV: Comma-separated, standard flat text file format.
- HDF5: Hierarchical data format, developed initially at the National Center for Supercomputing, is a fast and scalable storage format for numerical data, available in pandas using the PyTables library.
- Parquet: A binary, columnar storage format, part of the Apache Hadoop ecosystem, that provides efficient data compression and encoding and has been developed by Cloudera and Twitter. It is available for pandas through the pyarrow library, led by Wes McKinney, the original author of pandas.


It uses a test `DataFrame` that can be configured to contain numerical or text data, or both. For the HDF5 library, we test both the fixed and table format. The table format allows for queries and can be appended to. 

### Test Results

In short, the results are: 
- For purely numerical data, the HDF5 format performs best, and the table format also shares with CSV the smallest memory footprint at 1.6 GB. The fixed format uses twice as much space, and the parquet format uses 2 GB.
- For a mix of numerical and text data, parquet is significantly faster, and HDF5 uses its advantage on reading relative to CSV.

The notebook illustrates how to configure, test, and collect the timing using the `%%timeit` cell magic. At the same time demonstrates the usage of the related pandas commands required to use these storage formats.

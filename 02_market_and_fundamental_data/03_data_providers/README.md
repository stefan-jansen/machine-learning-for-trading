## 02 API access to market data

There are several options to access market data via API using Python.

### pandas Library

The notebook [01_datareader](01_datareader.ipynb) presents a few sources built into the pandas library. 
- The `pandas` library enables access to data displayed on websites using the read_html function 
- the related `pandas-datareader` library provides access to the API endpoints of various data providers through a standard interface 

### LOBSTER tick data

The notebook [02_lobster_itch_data](02_lobster_itch_data.ipynb) demonstrates the use of order book data made available by 
LOBSTER (Limit Order Book System - The Efficient Reconstructor), an [online](https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool that aims to provide easy-to-use, high-quality limit order book data.

Since 2013 LOBSTER acts as a data provider for the academic community, giving access to reconstructed limit order book data for the entire universe of NASDAQ traded stocks. More recently, it started offering a commercial service.

### Qandl

The notebook [03_quandl_demo](03_quandl_demo.ipynb) shows how Quandl uses a very straightforward API to make its free and premium data available. See [documentation](https://www.quandl.com/tools/api) for more details.

### zipline & Qantopian

The directory [04_zipline](04_zipline) contains the notebook [zipline_example](04_zipline/zipline_example.ipynb) that briefly introduces the backtesting library `zipline` that we will use later in the book from installation to data access and the implementation of a simple  trading strategy.


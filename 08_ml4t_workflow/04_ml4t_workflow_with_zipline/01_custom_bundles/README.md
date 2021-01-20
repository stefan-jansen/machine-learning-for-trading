# Zipline: Ingesting custom minute data

The `python` scripts in this directory sketch how to ingest custom minute data in Zipline. It is based on the Algoseek minute-bar trade data, which is not available for free. 
However, you can create a similar dataset by extracting the first, high, low, last and volume columns from the free sample of trade-and-quote data generously provided by [Algoseek](https://www.algoseek.com) [here](https://www.algoseek.com/ml4t-book-data.html).

Unfortunately, Zipline's pipeline API does not work for minute-bar data, so we are not using this custom bundle in the book but I am leaving this sample code here for adapation to your own prjects.

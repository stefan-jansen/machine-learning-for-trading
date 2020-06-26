# Custom Bundle for Japanese Stocks sourced from STOOQ

We are going to create a [custom bundle](https://www.Zipline.io/bundles.html#writing-a-new-bundle) for `Zipline` using Japanese equity data; see [download instructions](../../data/create_stooq_data.ipynb) first.  

We will take the following steps:
1. Create several data files containing information on tickers, prices, and adjustments
2. Code up a `Zipline` ingest function that handles the data processing and storage
3. Define a `Zipline` extension that registers the new `bundle`
4. Place the files in the `Zipline_ROOT` directory to ensure the `Zipline ingest` command finds them

## Setup

`Zipline` permits the creation of custom bundle containing open, high, low, close and volume (OHCLV) information, as well as adjustments like stock splits and dividend payments.

It stores the data per default a `.Zipline` directory in the user's home directory, `~/.Zipline`. However, you can modify the target location by setting the `Zipline_ROOT` environment variable as we do for the docker images provided with this book.   

## Data preprocessing

To prepare the data, we create three kinds of data tables in HDF5 format:
1. `equities`: contains a unique `sid`, the `ticker`, and a `name` for the security.
2. price tables with OHLCV data for each of the ~2,900 assets, named `jp.<sid>`
3. `splits`: contains split factors and is required; our data is already adjusted so we just add one line with a factor of 1.0 for one   

The file `stooq_preprocessing` implements these steps and produces the tables in the HDF5 file `stooq.h5`.

## `Zipline` ingest function

The file `stooq_jp_stocks.py` defines a function `stooq_jp_to_bundle(interval='1d')` that returns the `ingest` function required by `Zipline` to produce a custom bundle (see [docs](https://www.zipline.io/bundles.html#writing-a-new-bundle). It needs to have the following signature:

```python
ingest(environ,
       asset_db_writer,
       minute_bar_writer,
       daily_bar_writer,
       adjustment_writer,
       calendar,
       start_session,
       end_session,
       cache,
       show_progress,
       output_dir)
```

This function loads the information we crated in the previous step during the `ingest` process. It consists of a `data_generator()` that loads `(sid, ticker)` tuples as needed, and produces the corresponding OHLCV info in the correct format. It also adds information about the exchange so Zipline can associate the right calendar, and the range of trading dates.

It also loads the adjustment data, which in this case does not play an active role.

## Bundle registration

Zipline needs to know that the bundle exists and how to create the `ingest` function we just defined. To this end, we create an `extension.py` file that communicates the bundle's name, where to find the function that returns the `ingest` function (namely `stooq_jp_to_bundle()` in `stooq_jp_stocks.py`), and indicates the trading calendar to use (`XTKS` for Tokyo's exchange).

## File locations

Finally, we need to put these files in the right locations so that Zipline finds them. We can use symbolic links while keeping the actual files in this directory.

More specifically, we'll create symbolic links to 
1. to `stooq_jp_stocks.py` in the ZIPLINE_ROOT directory, and 
2. to stooq.h5 in `ZIPLINE_ROOT/custom_data`

In Linux or MacOSX, this implies opening the shell and running the following commands (where PROJECT_DIR refers to absolute path to the root folder of this repository on your machine)
```bash
cd $ZIPLINE_ROOT
ln -s PROJECT_DIR/11_decision_trees_random_forests/00_custom_bundle/stooq_jp_stocks.py
ln -s PROJECT_DIR/machine-learning-for-trading/11_decision_trees_random_forests/00_custom_bundle/extension.py .
mkdir custom_data
ln -s PROJECT_DIR/11_decision_trees_random_forests/00_custom_bundle/stooq.h5 custom_data/.
``` 

As a result, your directory structure should look as follows (some of these files will be symbolic links):
```python
ZIPLINE_ROOT
    |-extension.py
    |-stooq_jp_stocks.py
    |-custom_data
        |-stooq.h5
```



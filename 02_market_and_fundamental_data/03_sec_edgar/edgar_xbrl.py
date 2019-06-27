#!/usr/bin/env python
# coding: utf-8

# In[3]:


from io import BytesIO
from zipfile import ZipFile, BadZipFile
import requests
from datetime import date
from pathlib import Path
import pandas_datareader.data as web
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('fivethirtyeight')
data_path = Path('data')  # perhaps set to external harddrive to accomodate large amount of data

# Download FS & Notes

# The following code downloads and extracts all historical filings contained in the
# [Financial Statement and Notes](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html)
# (FSN) datasets for the given range of quarters:
# Downloads over 40GB of data!


SEC_URL = 'https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/'

today = pd.Timestamp(date.today())
this_year = today.year
this_quarter = today.quarter

past_years = range(2014, this_year)
filing_periods = [(y, q) for y in past_years for q in range(1, 5)]
filing_periods.extend([(this_year, q) for q in range(1, this_quarter + 1)])
for i, (yr, qtr) in enumerate(filing_periods, 1):
    print(yr, qtr, end=' ', flush=True)
    filing = f'{yr}q{qtr}_notes.zip'
    path = data_path / f'{yr}_{qtr}' / 'source'
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

    response = requests.get(SEC_URL + filing).content
    try:
        with ZipFile(BytesIO(response)) as zip_file:
            for file in zip_file.namelist():
                local_file = path / file
                if local_file.exists():
                    continue
                with local_file.open('wb') as output:
                    for line in zip_file.open(file).readlines():
                        output.write(line)
    except BadZipFile:
        continue

# Save to parquet
for f in data_path.glob('**/*.tsv'):
    file_name = f.stem + '.parquet'
    path = Path(f.parents[1]) / 'parquet'
    if (path / file_name).exists():
        continue
    if not path.exists():
        path.mkdir(exist_ok=True)
    try:
        df = pd.read_csv(f, sep='\t', encoding='latin1', low_memory=False)
    except:
        print(f)
    df.to_parquet(path / file_name)

# Metadata json

file = data_path / '2018_3' / 'source' / '2018q3_notes-metadata.json'
with file.open() as f:
    data = json.load(f)

pprint(data)

# Data Organization

# For each quarter, the FSN data is organized into eight file sets that contain information about
# submissions, numbers, taxonomy tags, presentation, and more.
# Each dataset consists of rows and fields and is provided as a tab-delimited text file:

# | File | Dataset      | Description                                                 |
# |------|--------------|-------------------------------------------------------------|
# | SUB  | Submission   | Identifies each XBRL submission by company, form, date, etc |
# | TAG  | Tag          | Defines and explains each taxonomy tag                      |
# | DIM  | Dimension    | Adds detail to numeric and plain text data                  |
# | NUM  | Numeric      | One row for each distinct data point in filing              |
# | TXT  | Plain Text   | Contains all non-numeric XBRL fields                        |
# | REN  | Rendering    | Information for rendering on SEC website                    |
# | PRE  | Presentation | Detail on tag and number presentation in primary statements |
# | CAL  | Calculation  | Shows arithmetic relationships among tags                   |

# Submission Data
# The latest submission file contains around 6,500 entries.


sub = pd.read_parquet(data_path / '2018_3' / 'parquet' / 'sub.parquet')
print(sub.info())

# Get AAPL submission
# The submission dataset contains the unique identifiers required to retrieve the filings:
# the Central Index Key (CIK) and the Accession Number (adsh). The following shows some of the information about Apple's 2018Q1 10-Q filing:

name = 'APPLE INC'
apple = sub[sub.name == name].T.dropna().squeeze()
key_cols = ['name', 'adsh', 'cik', 'name', 'sic', 'countryba', 'stprba',
            'cityba', 'zipba', 'bas1', 'form', 'period', 'fy', 'fp', 'filed']
print(apple.loc[key_cols])

# Build AAPL fundamentals dataset
# Using the central index key, we can identify all historical quarterly filings
# available for Apple, and combine this information to obtain 26 Forms 10-Q and nine annual Forms 10-K.
# Get filings
aapl_subs = pd.DataFrame()
for sub in data_path.glob('**/sub.parquet'):
    sub = pd.read_parquet(sub)
    aapl_sub = sub[(sub.cik.astype(int) == apple.cik) & (sub.form.isin(['10-Q', '10-K']))]
    aapl_subs = pd.concat([aapl_subs, aapl_sub])

# We find 15 quarterly 10-Q and 4 annual 10-K reports:
print(aapl_subs.form.value_counts())

# Get numerical filing data
# First, let's extract all numerical data available from the 19 Apple filings:
aapl_nums = pd.DataFrame()
for num in data_path.glob('**/num.parquet'):
    num = pd.read_parquet(num).drop('dimh', axis=1)
    aapl_num = num[num.adsh.isin(aapl_subs.adsh)]
    print(len(aapl_num))
    aapl_nums = pd.concat([aapl_nums, aapl_num])
aapl_nums.ddate = pd.to_datetime(aapl_nums.ddate, format='%Y%m%d')
aapl_nums.to_parquet(data_path / 'aapl_nums.parquet')

print(aapl_nums.info())

# Create P/E Ratio from EPS and stock price data

stock_split = 7
split_date = pd.to_datetime('20140604')
print(split_date)

# Filter by tag; keep only values measuring 1 quarter
eps = aapl_nums[(aapl_nums.tag == 'EarningsPerShareDiluted')
                & (aapl_nums.qtrs == 1)].drop('tag', axis=1)

# Keep only most recent data point from each filing
eps = eps.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))

# Adjust earnings prior to stock split downward
eps.loc[eps.ddate < split_date, 'value'] = eps.loc[eps.ddate < split_date, 'value'].div(7)
eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
eps = eps.rolling(4, min_periods=4).sum().dropna()

eps.plot(lw=2, figsize=(14, 6), title='Diluted Earnings per Share')
plt.xlabel('')
plt.savefig('diluted eps', dps=300);
plt.show()

symbol = 'AAPL.US'

aapl_stock = (web.
                  DataReader(symbol, 'quandl', start=eps.index.min())
                  .resample('D')
                  .last()
                  .loc['2014':eps.index.max()])
print(aapl_stock.info())

pe = aapl_stock.AdjClose.to_frame('price').join(eps.to_frame('eps'))
pe = pe.fillna(method='ffill').dropna()
pe['P/E Ratio'] = pe.price.div(pe.eps)
pe['P/E Ratio'].plot(lw=2, figsize=(14, 6), title='TTM P/E Ratio')
plt.show()

print(pe.info())

axes = pe.plot(subplots=True, figsize=(16, 8), legend=False, lw=2)
axes[0].set_title('Adj. Close Price')
axes[1].set_title('Diluted Earnings per Share')
axes[2].set_title('Trailing P/E Ratio')
plt.tight_layout()
plt.show()

# Explore Additional Fields
# The field `tag` references values defined in the taxonomy:

aapl_nums.tag.value_counts()

# Dividends per Share

fields = ['EarningsPerShareDiluted',
          'PaymentsOfDividendsCommonStock',
          'WeightedAverageNumberOfDilutedSharesOutstanding',
          'OperatingIncomeLoss',
          'NetIncomeLoss',
          'GrossProfit']

dividends = (aapl_nums
             .loc[aapl_nums.tag == 'PaymentsOfDividendsCommonStock', ['ddate', 'value']]
             .groupby('ddate')
             .mean())
shares = (aapl_nums
          .loc[aapl_nums.tag == 'WeightedAverageNumberOfDilutedSharesOutstanding', ['ddate', 'value']]
          .drop_duplicates()
          .groupby('ddate')
          .mean())
df = dividends.div(shares).dropna()
ax = df.plot.bar(figsize=(14, 5), title='Dividends per Share', legend=False)
ax.xaxis.set_major_formatter(mticker.FixedFormatter(df.index.strftime('%Y-%m')))
plt.show()

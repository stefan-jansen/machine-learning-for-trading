# Working with Market Data: NASDAQ_TotalView-ITCH Order Book

While FIX has a dominant large market share, exchanges also offer native protocols. The Nasdaq offers a TotalView ITCH direct data-feed protocol that allows subscribers to track individual orders for equity instruments from placement to execution or cancellation.

As a result, it allows for the reconstruction of the order book that keeps track of the list of active-limit buy and sell orders for a specific security or financial instrument. The order book reveals the market depth throughout the day by listing the number of shares being bid or offered at each price point. It may also identify the market participant responsible for specific buy and sell orders unless it is placed anonymously. Market depth is a key indicator of liquidity and the potential price impact of sizable market orders. 

In addition to matching market and limit orders, the Nasdaq also operates auctions or crosses that execute a large number of trades at market opening and closing. Crosses are becoming more important as passive investing continues to grow and traders look for opportunities to execute larger blocks of stock. TotalView also disseminates the Net Order Imbalance Indicator (NOII) for the Nasdaq opening and closing crosses and Nasdaq IPO/Halt cross.

> This example requires plenty of memory, likely above 16GB (I'm using 64GB and have not yet checked for the minimum requirement). If you run into capacity constraints, keep in mind that it is not essential for anything else in this book that you are able to run the code. First of all, it aims to demonstrate what kind of data you would be working with in an institutional investment context where the systems would have been built to manage data much larger than this single-day example. 

## Parsing Binary ITCH Messages

The ITCH v5.0 specification declares over 20 message types related to system events, stock characteristics, the placement and modification of limit orders, and trade execution. It also contains information about the net order imbalance before the open and closing cross.

The Nasdaq offers samples of daily binary files for several months. The GitHub repository for this chapter contains a notebook, build_order_book.ipynb that illustrates how to parse a sample file of ITCH messages and reconstruct both the executed trades and the order book for any given tick. 

The following table shows the frequency of the most common message types for the sample file used in the book (dated March 29, 2018). The code meanwhile updated to use a new sample from March 27, 2019.

| Message type | Order book impact                                                                  | Number of messages |
|:------------:|------------------------------------------------------------------------------------|-------------------:|
| A            | New unattributed limit order                                                       | 136,522,761        |
| D            | Order canceled                                                                     | 133,811,007        |
| U            | Order canceled and replaced                                                        | 21,941,015         |
| E            | Full or partial execution; possibly multiple messages for the same original order  | 6,687,379          |
| X            | Modified after partial cancellation                                                | 5,088,959          |
| F            | Add attributed order                                                               | 2,718,602          |
| P            | Trade Message (non-cross)                                                          | 1,120,861          |
| C            | Executed in whole or in part at a price different from the initial display price   | 157,442            |
| Q            | Cross Trade Message                                                                | 17,233             |

For each message, the specification lays out the components and their respective length and data types:


| Name                    | Offset  | Length  | Value      | Notes                                                                                |
|-------------------------|---------|---------|------------|--------------------------------------------------------------------------------------|
| Message Type            | 0       | 1       | S          | System Event Message                                                                 |
| Stock Locate            | 1       | 2       | Integer    | Always 0                                                                             |
| Tracking Number         | 3       | 2       | Integer    | Nasdaq internal tracking number                                                      |
| Timestamp               | 5       | 6       | Integer    | Nanoseconds since midnight                                                           |
| Order Reference Number  | 11      | 8       | Integer    | The unique reference number assigned to the new order at the time of receipt.        |
| Buy/Sell Indicator      | 19      | 1       | Alpha      | The type of order being added. B = Buy Order. S = Sell Order.                        |
| Shares                  | 20      | 4       | Integer    | The total number of shares associated with the order being added to the book.        |
| Stock                   | 24      | 8       | Alpha      | Stock symbol, right padded with spaces                                               |
| Price                   | 32      | 4       | Price (4)  | The display price of the new order. Refer to Data Types for field processing notes.  |
| Attribution             | 36      | 4       | Alpha      | Nasdaq Market participant identifier associated with the entered order               |

The notebooks [01_build_itch_order_book](01_parse_itch_order_flow_messages.ipynb), [02_rebuild_nasdaq_order_book](02_rebuild_nasdaq_order_book.ipynb) and [03_normalize_tick_data](03_normalize_tick_data.ipynb) contain the code to
- download NASDAQ Total View sample tick data,
- parse the messages from the binary source data
- reconstruct the order book for a given stock
- visualize order flow data
- normalize tick data

The code has been updated to use the latest NASDAQ sample file dated March 27, 2019.

Warning: the tick data is around 12GB in size and some processing steps can take several hours on a 4-core i7 CPU with 32GB RAM. 

## Regularizing tick data

The trade data is indexed by nanoseconds and is very noisy. The bid-ask bounce, for instance, causes the price to oscillate between the bid and ask prices when trade initiation alternates between buy and sell market orders. To improve the noise-signal ratio and improve the statistical properties, we need to resample and regularize the tick data by aggregating the trading activity.

We typically collect the open (first), low, high, and closing (last) price for the aggregated period, alongside the volume-weighted average price (VWAP), the number of shares traded, and the timestamp associated with the data.

The notebook [03_normalize_tick_data](03_normalize_tick_data.ipynb) illustrates how to normalize noisy tick using time and volume bars that use different aggregation methods.
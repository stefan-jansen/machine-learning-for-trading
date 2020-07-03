import sys
from pathlib import Path

sys.path.append(Path('~', '.zipline').expanduser().as_posix())
from zipline.data.bundles import register
from algoseek_1min_trades import algoseek_to_bundle
from datetime import time
from pytz import timezone
from trading_calendars import register_calendar
from trading_calendars.exchange_calendar_xnys import XNYSExchangeCalendar


class AlgoSeekCalendar(XNYSExchangeCalendar):
    """
    A calendar for trading assets before and after market hours

    Open Time: 4AM, US/Eastern
    Close Time: 19:59PM, US/Eastern
    """

    @property
    def name(self):
        """
        The name of the exchange that zipline
        looks for when we run our algorithm
        """
        return "AlgoSeek"

    @property
    def tz(self):
        return timezone("US/Eastern")

    open_times = (
        (None, time(4, 1)),
    )

    close_times = (
        (None, time(19, 59)),
    )


register_calendar(
        'AlgoSeek',
        AlgoSeekCalendar()
)

register('algoseek',
         algoseek_to_bundle(),
         calendar_name='AlgoSeek',
         minutes_per_day=960
         )



import logging
import pprint

from twisted.internet.task import LoopingCall
from scrapy import signals

logger = logging.getLogger(__name__)


class _LoopingExtension:
    def setup_looping_task(self, task, crawler, interval):
        self._interval = interval
        self._task = LoopingCall(task)
        crawler.signals.connect(self.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)

    def spider_opened(self):
        self._task.start(self._interval, now=False)

    def spider_closed(self):
        if self._task.running:
            self._task.stop()


class MonitorDownloadsExtension(_LoopingExtension):
    """
    Enable this extension to periodically log a number of active downloads.
    """

    def __init__(self, crawler, interval):
        self.crawler = crawler
        self.setup_looping_task(self.monitor, crawler, interval)

    @classmethod
    def from_crawler(cls, crawler):
        # fixme: 0 should mean NotConfigured
        interval = crawler.settings.getfloat("MONITOR_DOWNLOADS_INTERVAL", 10.0)
        return cls(crawler, interval)

    def monitor(self):
        active_downloads = len(self.crawler.engine.downloader.active)
        logger.info("Active downloads: {}".format(active_downloads))


class DumpStatsExtension(_LoopingExtension):
    """
    Enable this extension to log Scrapy stats periodically, not only
    at the end of the crawl.
    """

    def __init__(self, crawler, interval):
        self.stats = crawler.stats
        self.setup_looping_task(self.print_stats, crawler, interval)

    def print_stats(self):
        stats = self.stats.get_stats()
        logger.info("Scrapy stats:\n" + pprint.pformat(stats))

    @classmethod
    def from_crawler(cls, crawler):
        interval = crawler.settings.getfloat("DUMP_STATS_INTERVAL", 60.0)
        # fixme: 0 should mean NotConfigured
        return cls(crawler, interval)

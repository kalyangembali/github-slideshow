import os
import yfinance as yf
import pandas as pd
from loguru import logger
from config import settings

class YFinanceFetcher:
    def __init__(self, start_date=settings.TESTING_START_DATE, cache_dir=settings.CACHE_DIR):
        self.start_date = start_date
        self.cache_dir = cache_dir

    def _get_cache_path(self, symbol: str, interval: str) -> str:
        safe_symbol = symbol.replace("^", "").replace("=", "_")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{interval}.parquet")

    def fetch_symbol(self, symbol: str, interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
        """Fetch data for a single symbol, with parquet caching."""
        cache_path = self._get_cache_path(symbol, interval)

        if not force_refresh and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded {symbol} from cache ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {symbol}: {e}. Redownloading...")

        logger.info(f"Downloading {symbol} ({interval}) from {self.start_date}...")
        try:
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                # Intraday data limits in yfinance (max 60 days for 5m)
                df = yf.download(symbol, period="60d", interval=interval, progress=False)
            else:
                df = yf.download(symbol, start=self.start_date, interval=interval, progress=False)

            # yfinance sometimes returns MultiIndex columns if one ticker is passed, flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Ensure 'Date' or timestamp is index
            df.index.name = 'Date'

            # Save to cache
            df.to_parquet(cache_path)
            logger.info(f"Saved {symbol} to cache ({len(df)} rows)")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all(self, symbols: list = settings.ALL_SYMBOLS, interval: str = "1d", force_refresh: bool = False):
        """Fetch data for all symbols in the universe."""
        data_dict = {}
        for sym in symbols:
            df = self.fetch_symbol(sym, interval=interval, force_refresh=force_refresh)
            if not df.empty:
                data_dict[sym] = df
        return data_dict

if __name__ == "__main__":
    fetcher = YFinanceFetcher(start_date="2023-01-01")
    # Fetch a small subset for testing
    test_symbols = ["RELIANCE.NS", "TCS.NS", "^NSEI"]
    data = fetcher.fetch_all(symbols=test_symbols, force_refresh=True)
    print(f"Fetched {len(data)} symbols successfully.")

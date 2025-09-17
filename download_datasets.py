import requests
import pandas as pd
import yfinance as yf
import os
import shutil
from investiny import historical_data


ALPHA_VANTAGE_API_KEY = "sjlkfasjfsouut"
START_DATE="2010-01-01"
END_DATE="2020-12-31"
TRAINING_END_DATE="2017-12-31"
TESTING_START_DATE="2018-01-01"

START_DATE_INVESTINY="01/01/2010"
END_DATE_INVESTINY="12/31/2020"
TRAINING_END_DATE_INVESTINY="12/31/2017"
TESTING_START_DATE_INVESTINY="01/01/2018"

# cache the US trading calendar once (S&P 500 dates)
_REF_INDEX = None

def get_us_calendar():
    """Return DatetimeIndex of US trading days (from ^GSPC)."""
    global _REF_INDEX
    if _REF_INDEX is None:
        ref = yf.download("^GSPC", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if ref.empty:
            raise RuntimeError("Could not build US trading calendar from ^GSPC.")
        _REF_INDEX = ref.index.tz_localize(None)   # <-- IMPORTANT: return index
    return _REF_INDEX

def reset(base_dir="datasets"):
    """
    Reset the datasets folder:
    1. Delete the datasets folder if it exists
    2. Create datasets/
    3. Create datasets/training_data
    4. Create datasets/testing_data
    """
    # 1. Delete datasets folder if exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"🗑️ Removed existing {base_dir}/")

    # 2. Create datasets folder
    os.makedirs(base_dir, exist_ok=True)
    print(f"📂 Created {base_dir}/")

    # 3. Create training_data folder
    train_path = os.path.join(base_dir, "training_data")
    os.makedirs(train_path, exist_ok=True)
    print(f"📂 Created {train_path}/")

    # 4. Create testing_data folder
    test_path = os.path.join(base_dir, "testing_data")
    os.makedirs(test_path, exist_ok=True)
    print(f"📂 Created {test_path}/")

    return train_path, test_path



def download_with_yfinance(ticker, name, base_dir="datasets", fx_ticker=None):
    """
    Download data from Yahoo Finance, optionally convert to USD,
    split into training (2010-2017) and testing (2018-2020), and save to CSVs.

    Parameters
    ----------
    ticker : str
        The Yahoo Finance ticker for the index/asset.
    name : str
        The name to use for saving CSVs.
    base_dir : str
        Directory where training/testing_data subfolders exist.
    fx_ticker : str or None
        Yahoo Finance FX ticker (quoted as USD per 1 unit of local currency).
        Examples:
            - "GBPUSD=X" for FTSE
            - "JPYUSD=X" for Nikkei
            - None for USD-denominated assets (S&P, EEM, Gold, TNX)
    """
    try:
        # 1. Download full dataset
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        # If still empty -> error handling
        if df.empty:
            print(f"❌ Failed to download data for {ticker} ({name}). No data found.")
            return

        # Ensure index is datetime
        df.index = df.index.tz_localize(None)

        # 2) pick the adjusted close series (with auto_adjust=True, 'Close' is adjusted)
        series = df["Close"][ticker].copy()

        # 3) align to US trading calendar and forward-fill missing days
        ref_index = get_us_calendar()
        aligned = series.reindex(ref_index).ffill()

        # 4) FX conversion (if fx_ticker provided)
        if fx_ticker:
            fx_df = yf.download(fx_ticker, start=START_DATE, end=END_DATE,
                                auto_adjust=True, progress=False)
            if fx_df.empty:
                print(f"❌ FX download failed for {fx_ticker}; skipping conversion for {name}.")
                return

            fx_df.index = pd.to_datetime(fx_df.index).tz_localize(None)
            fx_series = (fx_df["Close"]
                         if not isinstance(fx_df.columns, pd.MultiIndex)
                         else fx_df["Close"].iloc[:, 0]).copy()
            fx_aligned = fx_series.reindex(ref_index).ffill()

            print(fx_aligned.head())

            aligned = aligned * fx_aligned  # Convert local index to USD        

        # 5) drop any leading NaNs (before the asset’s first real price)
        first_valid = aligned.first_valid_index()
        if first_valid is None:
            print(f"⚠️ {ticker} ({name}) has no valid data after alignment.")
            return
        aligned = aligned.loc[first_valid:]

        # 6) rebuild minimal DataFrame to save
        out = aligned.to_frame(name="Close")




        # 7) split into train/test
        train_df = out.loc[START_DATE:TRAINING_END_DATE]
        test_df  = out.loc[TESTING_START_DATE:END_DATE]

        # 8) Save to CSV
        train_path = os.path.join(base_dir, "training_data", f"{name}.csv")
        test_path = os.path.join(base_dir, "testing_data", f"{name}.csv")

        train_df.to_csv(train_path)
        test_df.to_csv(test_path)

        print(f"✅ Saved training data from Yahoo Finance -> {train_path} ({len(train_df)} rows)")
        print(f"✅ Saved testing data from Yahoo Finance -> {test_path} ({len(test_df)} rows)")

    except Exception as e:
        print(f"❌ Error downloading {ticker} ({name}): {e}")

def download_gold_from_investing(ticker, name, base_dir="datasets"):
    """
    Gold spot (XAU/USD) via investiny (ID=8830), aligned to US calendar, ffilled,
    split 2010–2017 / 2018–2020, saved to CSVs.
    """
    try:
        # --- Fetch XAU/USD (Investing.com symbol id = 8830) ---
        # Some versions use 'id', others 'symbol_id'. Try id first.
        payload = historical_data(
            investing_id=8830,                   # XAU/USD
            from_date=START_DATE_INVESTINY,        # e.g., "01/01/2010"
            to_date=END_DATE_INVESTINY,            # e.g., "12/31/2020"
            interval="D"
        )

        if not payload:
            print("❌ No data returned for XAU/USD.")
            return
        
        

        # Convert payload -> DataFrame  (expects keys: date (ms), close, etc.)
        df = pd.DataFrame(payload)
        if df.empty or "date" not in df or "close" not in df:
            print("❌ Unexpected payload for XAU/USD from Investing.")
            return

        df["date"] = pd.to_datetime(df["date"])
        
        df.set_index("date", inplace=True)
        series = df["close"].astype(float)
        
        # Align to US calendar & forward-fill
        ref_index = get_us_calendar()
        aligned = series.reindex(ref_index).ffill()

        

        fv = aligned.first_valid_index()
        if fv is None:
            print("⚠️ Gold XAU/USD has no valid data after alignment.")
            return
        aligned = aligned.loc[fv:]

        out = aligned.to_frame(name="Close")

        

        # Split and save
        train_df = out.loc[START_DATE_INVESTINY:TRAINING_END_DATE_INVESTINY]
        test_df  = out.loc[TESTING_START_DATE_INVESTINY:END_DATE_INVESTINY]

        train_path = os.path.join(base_dir, "training_data", f"{name}.csv")
        test_path  = os.path.join(base_dir, "testing_data", f"{name}.csv")

        train_df.to_csv(train_path)
        test_df.to_csv(test_path)

        print(f"✅ Saved training data from investing.com -> {train_path} ({len(train_df)} rows)")
        print(f"✅ Saved testing data from investing.com -> {test_path} ({len(test_df)} rows)")

    except Exception as e:
        print(f"❌ Error downloading Gold (XAU/USD) from Investing.com: {e}")

def main():

    #0. Reset
    reset()

    #1. Download S&P 500 index (^GSPC)
    snp_ticker = "^GSPC"
    download_with_yfinance(snp_ticker,"snp_500")    

    #2. Download FTSE 100 Index (^FTSC)
    ftse_ticker = "^FTSC"
    download_with_yfinance(ftse_ticker,"ftse_100", fx_ticker="GBPUSD=X")    

    #3. Download Nikkei 225 Index (^N225)
    n225_ticker = "^N225"
    download_with_yfinance(n225_ticker,"n_225", fx_ticker="JPYUSD=X")    

    #4. Download MSCI Emerging Markets ETF (^EEM)
    eem_ticker = "EEM"
    download_with_yfinance(eem_ticker,"eem")    

    #5. Download gold index (XAUUSD=X)
    # gold_ticker = "GC=F"
    # download_with_yfinance(gold_ticker,"gold")
    gold_ticker = "XAU/USD"
    download_gold_from_investing(gold_ticker,"gold")

    #6. Download US 10Y Treasury bond (^TNX)
    tnx_ticker = "^TNX"
    download_with_yfinance(tnx_ticker,"tnx")    


if __name__ == "__main__":
    main()
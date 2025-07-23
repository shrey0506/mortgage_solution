import requests
import pandas as pd
from bs4 import BeautifulSoup
from mortgage.src.utils.logger import logger

def fetch_latest_interest_rate() -> float:
    """
    Fetches the latest base interest rate from the Bank of England website.

    Returns:
        float: Latest interest rate, or fallback if parsing fails.
    """
    url = "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    fallback_rate = 4.25  # Known latest rate as of July 2025

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info("Successfully accessed Bank of England interest rate page.")
    except requests.RequestException as e:
        logger.warning(f"Request failed, using fallback rate: {e}")
        return fallback_rate

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    if not table:
        logger.warning("Interest rate table not found, using fallback rate.")
        return fallback_rate

    rows = []
    for tr in table.find_all("tr"):
        cols = tr.find_all("td")
        cols = [td.text.strip() for td in cols]
        if cols:
            rows.append(cols)

    if not rows:
        logger.warning("No data rows found in the table, using fallback rate.")
        return fallback_rate

    df = pd.DataFrame(rows, columns=["Date", "Interest Rate"])
    df['Date'] = pd.to_datetime(df["Date"], format="%d %b %Y", errors='coerce')
    df = df.dropna(subset=["Date"])

    if df.empty:
        logger.warning("Date parsing failed, using fallback rate.")
        return fallback_rate

    # Assuming the latest rate is the first one in the table after sorting by date
    latest_rate_row = df.sort_values(by="Date", ascending=False).iloc[0]
    latest_rate = latest_rate_row["Interest Rate"]

    # Clean and validate the interest rate string before conversion
    latest_rate_str = str(latest_rate).replace('%', '').strip()

    try:
        latest_rate_float = float(latest_rate_str)
        logger.info(f"Latest interest rate: {latest_rate_float}%")
        return latest_rate_float
    except ValueError:
        logger.warning(f"Failed to convert rate '{latest_rate_str}' to float, using fallback rate.")
        return fallback_rate
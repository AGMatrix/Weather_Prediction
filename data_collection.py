"""
Lab 2 - Weather Prediction
Module for collecting CF6 weather reports from National Weather Service.
"""
import requests
from datetime import datetime, timedelta
import time
import os #for handling directories
import re
import ssl
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Ignore SSL certificate errors for urllib
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE #Ignoring the risk it involves as it is working in a testing environment
# ctx.load_verify_locations()

def construct_cf6_url(office_code, city_code, year, month):
    """
    Constructs the URL for a CF6 report.

    Args:
        office_code (str): Three-letter code for the NWS office
        city_code (str): Three-letter code for the city
        year (int): Year for the report
        month (int): Month for the report (1-12)

    Returns:
        str: URL for the CF6 report
    """
    # Updated URL pattern based on current NWS website structure
    return f"https://forecast.weather.gov/product.php?site=NWS&issuedby={city_code}&product=CF6&format=CI&version=1&glossary=0"


def fetch_cf6_report_bs4(city_code, year, month):
    """
    Fetches a CF6 report from the National Weather Service website using BeautifulSoup.

    Args:
        city_code (str): Three-letter code for the city
        year (int): Year for the report
        month (int): Month for the report (1-12)

    Returns:
        str: The text content of the CF6 report, or None if not found
    """
    try:
        # First, try to find CF6 reports by browsing the main NWS product page
        base_url = "https://forecast.weather.gov/product_types.php?site=NWS&product=CF6"
        print(f"  Fetching list of CF6 reports from {base_url}")

        #In this case- it fails
        html = urlopen(base_url, context=ctx).read()
        soup = BeautifulSoup(html, "html.parser")

        # Look for links that might contain CF6 reports for our city
        for link in soup.find_all('a'):#looking for hyperlinks
            href = link.get('href', '')
            link_text = link.get_text()#converting output to text

            if city_code in link_text or city_code in href:
                report_url = "https://forecast.weather.gov/" + href if href.startswith('/') else href
                print(f"  Found potential CF6 link for {city_code}: {report_url}")

                try:
                    report_html = urlopen(report_url, context=ctx).read()
                    report_soup = BeautifulSoup(report_html, "html.parser")

                    pre_tag = report_soup.find('pre')#contains the raw text of the CF6 weather report
                    if pre_tag and "PRELIMINARY LOCAL CLIMATOLOGICAL DATA" in pre_tag.get_text():
                        cf6_text = pre_tag.get_text()
                        print(f"Found CF6 data for {city_code}")
                        return cf6_text
                except Exception as e:
                    print(f"Error fetching report from link: {e}")
                    continue

        # If we couldn't find the report through browsing, try direct URL access
        direct_url = construct_cf6_url(None, city_code, year, month)
        print(f"Trying direct URL: {direct_url}")

        response = requests.get(direct_url)
        if response.status_code == 200:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            pre_tag = soup.find('pre')

            if pre_tag and "PRELIMINARY LOCAL CLIMATOLOGICAL DATA" in pre_tag.get_text():
                cf6_text = pre_tag.get_text()
                print(f"Found CF6 data for {city_code} through direct URL")
                return cf6_text

        print(f"Could not find CF6 data for {city_code}")
        return None
    except Exception as e:
        print(f"Error in fetch_cf6_report_bs4 for {city_code}: {e}")
        return None


def fetch_cf6_report(office_code, city_code, year, month):
    """
    Fetches a CF6 report from the National Weather Service website.
    Tries multiple approaches to find the data.

    Args:
        office_code (str): Three-letter code for the NWS office
        city_code (str): Three-letter code for the city
        year (int): Year for the report
        month (int): Month for the report (1-12)

    Returns:
        str: The text content of the CF6 report, or None if not found
    """
    print(f"Fetching data for {city_code}...")

    # Try BeautifulSoup approach first
    cf6_text = fetch_cf6_report_bs4(city_code, year, month)
    if cf6_text:
        # print("bs4")for debugging -  this approach works
        return cf6_text #Tree-based cleaner HTML Parsing

    # If BeautifulSoup approach failed, try direct request with regex - However, BS is successful in this case.
    url = construct_cf6_url(office_code, city_code, year, month)
    try:
        response = requests.get(url)
        print(f"  URL: {url}")
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            # Extract the CF6 report from the HTML
            text = response.text

            # Look for the CF6 data which is usually between the header and footer
            # It often starts with PRELIMINARY LOCAL CLIMATOLOGICAL DATA
            pattern = r'PRELIMINARY LOCAL CLIMATOLOGICAL DATA[\s\S]*?={80,}'#PRELIMINARY LOCAL CLIMATOLOGICAL DATA followed by any content until a line of 80+ =
            match = re.search(pattern, text, re.DOTALL)#CF6 reports span multiple lines. Without re.DOTALL, regex would stop matching at the first newline.

            if match:
                return match.group(0)
            else:
                # Try alternative pattern looking for the CF6 code and the structured data
                pattern = r'CF6' + city_code + r'[\s\S]*?={80,}[\s\S]*?={80,}' #looks for CF6ROC (for Rochester) between two sections of 80+ = characters.
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return match.group(0)#group(0) is always the entire match. Other groups return captured groups (if parentheses () are used in the pattern).
                else:
                    print(f"Could not extract CF6 data from the HTML for {city_code}")
                    # Save the first 1000 chars of the response for debugging
                    with open(f"debug_{city_code}_{year}_{month:02d}.html", "w") as f:
                        f.write(text[:1000])
                    return None
        else:
            print(f"Failed to fetch CF6 report for {city_code} ({year}-{month:02d}): HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching CF6 report for {city_code} ({year}-{month:02d}): {e}")
        return None


def collect_historical_cf6_data(city_mappings, start_date, end_date, output_dir="data", use_mock_data=False):
    """
    Collects historical CF6 data for multiple cities over a specified time range.

    Args:
        city_mappings (dict): Dictionary mapping city codes to their NWS office codes
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
        output_dir (str): Directory to save the fetched reports
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True) #doesn't give error if it exists

    print(f"Starting data collection from {start_date} to {end_date}")
    print(f"Collecting data for cities: {list(city_mappings.keys())}")

    # Count successful downloads
    successful_downloads = 0
    failed_downloads = 0

    # Iterate through each month in the date range
    current_date = start_date.replace(day=1) #Ensures iteration starts from the first day of the month.
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month

        print(f"Processing {year}-{month:02d}...")

        for city_code, office_code in city_mappings.items():
            # Fetch the CF6 report
            cf6_report = fetch_cf6_report(office_code, city_code, year, month)

            if cf6_report:
                # Save the report to a file and then store it in a directory
                filename = f"{city_code}_{year}_{month:02d}.txt"
                with open(os.path.join(output_dir, filename), "w") as f:
                    f.write(cf6_report)

                print(f"Saved CF6 report for {city_code} ({year}-{month:02d})")
                successful_downloads += 1
            else:
                print(f"Failed to get CF6 report for {city_code} ({year}-{month:02d})")
                failed_downloads += 1

            # Be nice to the server and avoid rate limiting
            time.sleep(1)

        # Move to the next month or year
        if month == 12:
            current_date = current_date.replace(year=year + 1, month=1)
        else:
            current_date = current_date.replace(month=month + 1)

    print(f"Data collection complete. Successfully downloaded {successful_downloads} reports, failed to download {failed_downloads} reports.")
    #in this case- Successfully downloaded 1197 reports, failed to download 0 reports.

def collect_daily_cf6_data(city_mappings, date, output_dir="daily_data"):
    """
    Collects daily CF6 data for multiple cities for a specific date.
    This is useful for collecting recent data for prediction.

    Args:
        city_mappings (dict): Dictionary mapping city codes to their NWS office codes
        date (datetime): Date for data collection
        output_dir (str): Directory to save the fetched reports
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    year = date.year
    month = date.month
    day = date.day

    print(f"Collecting daily data for {date.strftime('%Y-%m-%d')}")

    for city_code, office_code in city_mappings.items():
        # For daily data, we'll use the same approach as for monthly data
        # The NWS website typically has the latest data available
        cf6_report = fetch_cf6_report(office_code, city_code, year, month)

        if cf6_report:
            # Save the report to a file
            filename = f"{city_code}_{year}_{month:02d}_{day:02d}.txt"
            with open(os.path.join(output_dir, filename), "w") as f:
                f.write(cf6_report)

            #Tracks success/failure metrics for debugging
            print(f"Saved daily CF6 report for {city_code} ({year}-{month:02d}-{day:02d})")
        else:
            print(f"Failed to get daily CF6 report for {city_code} ({year}-{month:02d}-{day:02d})")

        # Be nice to the server and avoid rate limiting
        time.sleep(1)


if __name__ == "__main__":
    # Example usage:

    # Define city mappings (city code -> NWS office code)
    # These are some cities in and around the Northeast/Great Lakes
    city_mappings = {
        "ROC": "BUF",  # Rochester, NY
        "BUF": "BUF",  # Buffalo, NY
        "SYR": "BGM",  # Syracuse, NY
        "ALB": "ALY",  # Albany, NY
        "BOS": "BOX",  # Boston, MA
        "DTW": "DTX",  # Detroit, MI
        "CLE": "CLE",  # Cleveland, OH
        "PIT": "PBZ",  # Pittsburgh, PA
        "ERI": "CLE",  # Erie, PA
        "ORD": "LOT",  # Chicago, IL
        "MSP": "MPX",  # Minneapolis, MN
        "BTV": "BTV",  # Burlington, VT
        "PWM": "GYX",  # Portland, ME
        "BDL": "BOX",  # Hartford, CT
        "PHL": "PHI",  # Philadelphia, PA
        "IAD": "LWX",  # Washington, DC area
        "CVG": "ILN",  # Cincinnati, OH
        "IND": "IND",  # Indianapolis, IN
        "MKE": "MKX",  # Milwaukee, WI
    }

    start_date = datetime(2020, 1, 1).date()
    end_date = datetime(2025, 3, 31).date()
    collect_historical_cf6_data(city_mappings, start_date, end_date)

    # Collect recent daily data for prediction
    today = datetime.now().date()
    first_of_month = today.replace(day=1)  # Get the 1st day of current month
    daily_output_dir = "daily_data"
    # Iterate through all days from 1st of month to yesterday
    current_date = first_of_month
    while current_date <= today:
        collect_daily_cf6_data(city_mappings, current_date, daily_output_dir)
        current_date += timedelta(days=1)
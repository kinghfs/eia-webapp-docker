import io
import os
import re
import pandas as pd
import datetime as dt

import asyncio
import aiohttp
import requests
from urllib.parse import urljoin

from typing import Optional

from time import perf_counter


EIA_BASE_URL = "https://www.eia.gov"
EIA_ARCHIVE_URI = "/petroleum/supply/weekly/archive/"
EIA_REPORT_TABLE_URI = "/csv/table1.csv"

PETROLEUM_STOCK_INDEX_TOP_LEVEL = [
    "Crude Oil", 
    "Total Motor Gasoline", 
    "Fuel Ethanol", 
    "Kerosene-Type Jet Fuel", 
    "Distillate Fuel Oil",
    "Residual Fuel Oil", 
    "Propane/Propylene",
    "Other Oils",
    "Unfinished Oils", 
    "Total Stocks (Including SPR)", 
    "Total Stocks (Excluding SPR)"
]

def get_report_urls(earliest: Optional[dt.datetime] = None) -> list[str]:
    '''Generate EIA archive URLs for WPSRs
    '''
    resp = requests.get(urljoin(EIA_BASE_URL, EIA_ARCHIVE_URI))
    
    if resp.status_code != 200:
        
        raise Exception(f"Failed to retrieve report urls with status code {resp.status_code}")
        
    contents = resp.content.decode(encoding='cp1252')

    report_pattern = re.compile("(/petroleum/supply/weekly/archive/[0-9]{4}/[0-9]{4}_[0-9]{1,2}_[0-9]{1,2}(?:_data)?)")
    
    report_uris = report_pattern.findall(contents)
    
    if isinstance(earliest, dt.datetime):
        
        recent_reports = []
        
        for uri in report_uris:
            
            report_date = dt.datetime.strptime(uri.split('/')[-1].replace('_data', ''), '%Y_%m_%d')
            
            if report_date >= earliest:
                
                recent_reports.append(uri)
                
        report_uris = recent_reports
    
    urls = [urljoin(EIA_BASE_URL, uri) + EIA_REPORT_TABLE_URI for uri in report_uris]
    
    return urls
    
async def download_table(session, url, semaphore) -> tuple[str, pd.DataFrame]:
    '''Send a request, throttling with Semaphore
    '''
    async with semaphore, session.get(url) as response:
        
        with io.StringIO(await response.text(encoding='cp1252')) as text_io:
            
            return url, pd.read_csv(text_io, on_bad_lines='skip')

async def download_many_tables(urls, concur_req):
    '''Asynchronously download all csv files from urls
    '''
    df = pd.DataFrame(columns=PETROLEUM_STOCK_INDEX_TOP_LEVEL)
    
    semaphore = asyncio.Semaphore(concur_req)

    async with aiohttp.ClientSession() as session:
        
        to_do = [download_table(session, url, semaphore)
                 for url in urls]
        
        to_do_itr = asyncio.as_completed(to_do)

        for coro in to_do_itr:
            
            this_url, this_df = await coro
            
            try: # normalise table format
                this_df = format_report_table(this_df)
                this_df['Endpoint'] = [this_url]
                
            except KeyError: # invalid table format
                print(f"Failed to extract table from {this_url}")
                pass
                
            else: # merge dataframes
                
                df = this_df.copy() if df.empty else pd.concat([df, this_df], axis=0)
               
        df.index = pd.to_datetime(df.index, format='%m/%d/%y')
    
        return df

def format_report_table(df: pd.DataFrame) -> dict:
    """Extracts the top level stock inventories from a Petroleum Stocks DataFrame
    """
    # Remove second-level stock rows
    summary_df = df[df['STUB_1'].isin(PETROLEUM_STOCK_INDEX_TOP_LEVEL)]
    
    # Get latest stock levels (first column after STUB_1)
    stock_s = summary_df.set_index('STUB_1').iloc[:, 0]
    
    stock_s.index.name = None
    
    # Convert str values to float
    stock_s = stock_s.str.replace(',', '').astype(float)
    
    # Reformat as df row
    stock_df = stock_s.to_frame().T
    
    # Add SPR column
    stock_df['SPR'] = (stock_df['Total Stocks (Including SPR)']
                       - stock_df['Total Stocks (Excluding SPR)'])
    
    return stock_df[PETROLEUM_STOCK_INDEX_TOP_LEVEL + ['SPR']]
    
def save_inventory_table_to_file(df, cache_file:Optional[str]=None):
    '''Save inventory dataframe to file
    '''
    if cache_file is None:
        cache_file = 'reports.csv'

    df.to_csv(cache_file)

def download_eia_reports(urls, concur_req:int = 10):
    '''Function to run coroutines that download WPRS reports
    '''
    coro = download_many_tables(urls, concur_req)
    
    df = asyncio.run(coro)
    
    return df

def load_cached_reports(cache_file) -> pd.DataFrame:
    # check valid path here
    if os.path.isfile(cache_file):
        print("Cache found")
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.DataFrame()
    
    return df


def add_latest_report_if_new(old_df: pd.DataFrame()) -> pd.DataFrame: 
    # Check for new release (non-archive)
    latest_url = 'https://ir.eia.gov/wpsr/table1.csv'
    latest_df = download_eia_reports([latest_url], concur_req=1)
    
    if latest_df.index[-1] != old_df.index[-1]:
        # New release
        df = latest_df.copy() if old_df.empty else pd.concat([old_df, latest_df], axis=0)
    else:
        print("Latest release already cached")
        df = old_df

    return df


def download_wpsr_since_date(date: dt.datetime, cache_file:Optional[str]=None) -> pd.DataFrame:
    # Get valid report endpoints
    report_urls = get_report_urls(earliest=date)

    # Load cached reports
    df = load_cached_reports(cache_file)

    # Ignore cached urls
    if not df.empty:
        cached_urls = df['Endpoint'].to_list()
        report_urls = list(set(report_urls) - set(cached_urls))

    # Download un-cached archive reports
    new_df = download_eia_reports(report_urls, concur_req=10)

    # Merge new reports and cache
    if not new_df.empty:
        df = new_df.copy() if df.empty else pd.concat([df, new_df], axis=0)
        df = df.sort_index()

    # Check latest report
    df = add_latest_report_if_new(df)

    # Cache results
    save_inventory_table_to_file(df)
    return df

    
if __name__ == '__main__':

    start_t = perf_counter()

    cache_file = os.path.join(os.getcwd(), 'reports.csv')
    
    cutoff =  dt.datetime.now() - dt.timedelta(weeks=104)

    df = download_wpsr_since_date(cutoff, cache_file=cache_file)
    
    print(f"Finished scraping EIA WPSRs")
    print(f"{len(df)} reports found")
    seconds = perf_counter() - start_t
    print(f"That took {seconds//60:.0f}mins, {seconds%60:.2f}seconds", flush=True)


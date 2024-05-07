import base64
from io import BytesIO

import numpy as np
import pandas as pd

import pytz
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

import yfinance as yf

from flask import Flask

from async_scrape_eia import download_wpsr_since_date


wall_st = pytz.timezone("America/New_York")
mdate_fmt = mdates.DateFormatter('%m/%y')
date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)

def load_inventory_data():
    df = pd.read_csv('./reports.csv', index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(wall_st)
    df = df.sort_index()
    return df

def add_inventory_features(df):
    df['Commercial Crude (Excluding SPR)'] = df['Crude Oil'] - df['SPR']
    df['Weekly Commercial Draw/Build'] = df['Commercial Crude (Excluding SPR)'].diff()
    df['Weekly Commercial Draw/Build Rolling'] = df['Weekly Commercial Draw/Build'].rolling(4).sum()
    return df

def get_oil_data(days=365):
    wti = yf.Ticker("CL=F")
    years = int(np.ceil(days/365)) + 1
    hist = wti.history(period=f"{years}y")
    return hist

def plot_oil_price(df, ax, days=365):
    sma_freq = 20
    consider_df = df.tail(days+sma_freq)

    ax.set_title("WTI Oil Price")
    ax.plot(consider_df.index[20:], consider_df['Close'].tail(days))
    ax.plot(consider_df.index[20:], consider_df['Close'].rolling(sma_freq).mean().tail(days), label="SMA (20-day)")
    ax.legend()
    ax.set_ylabel("Price per barrel ($)")
    ax.xaxis.set_major_formatter(mdate_fmt)

def plot_draw_build_bar(df, ax, weeks=24):
    # Draw/Build
    consider_df = df.tail(weeks)
    builds = consider_df[consider_df['Weekly Commercial Draw/Build']>=0]
    draws = consider_df[consider_df['Weekly Commercial Draw/Build']<0]
    
    ax.set_title("Commercial Inventory Weekly Change")
    ax.bar(builds.index, builds['Weekly Commercial Draw/Build'], color='red', width=1.5, alpha=0.5, label="Build")
    ax.bar(draws.index, draws['Weekly Commercial Draw/Build'], color='green', width=1.5, alpha=0.5, label="Draw")
    # ax.plot(consider_df.index, consider_df['Weekly Commercial Draw/Build Rolling'], color='black', alpha=0.5)
    ax.legend()
    ax.set_ylabel("Change (Mb)")
    ax.xaxis.set_major_formatter(mdate_fmt)
    # ax.xaxis.set_major_locator(date_locator)

def plot_draw_build_dist(df, ax, weeks=24):
    # Boxplot of Draw/Build
    consider_df = df.tail(weeks)
    last_stock_change = consider_df['Weekly Commercial Draw/Build'].iloc[-1]
    
    ax.set_title("Commercial Inventory Change Distribution")
    ax.boxplot(consider_df['Weekly Commercial Draw/Build'].iloc[:-1],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue'),
                      flierprops=dict(marker = "s", markerfacecolor = "red"))

    if last_stock_change > 0:
        color = 'red'
    else:
        color = 'green'
    ax.hlines(last_stock_change, 
                     xmin=0.5, xmax=1.5, 
                     color=color, alpha=0.5, 
                     label="Last Reported Change") 
    
    ax.xaxis.set_ticklabels([])
    ax.legend()

def plot_spr_line(df, ax, weeks=104, references=dict()):
    # SPR Level
    consider_df = df.tail(weeks)
    ax.set_title("SPR Inventory")
    ax.plot(consider_df.index, consider_df['SPR'], color='blue', alpha=0.5)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(references)))
    i = 0
    for label, lookback in references.items():
        less_df = df.loc[:(df.index.max()-lookback)].iloc[-1]
        ax.hlines(y=less_df['SPR'],
                  xmin=consider_df.index.min(),
                  xmax=consider_df.index.max(),
                  linewidth=2, alpha=0.3, color=colors[i],
                  label=label)
        i += 1

    ylabels = [ int(l) for l in np.linspace(consider_df['SPR'].min(), consider_df['SPR'].max(), 5) ]
    ax.yaxis.set_ticks(ylabels)
    ax.set_ylabel("Total (Mb)")
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(mdate_fmt)
    if references:
        ax.legend()

def plot_commercial_inventory_line(df, ax, weeks=104, references:dict[str, dt.timedelta]=dict()):
    # Commercial Stock Level
    consider_df = df.tail(weeks)
    ax.set_title("Commercial Crude Inventory")
    ax.plot(consider_df.index, consider_df['Commercial Crude (Excluding SPR)'], color='blue', alpha=0.2)
    ax.plot(consider_df.index, consider_df['Commercial Crude (Excluding SPR)'].rolling(4).mean(),
                   color='k', alpha=0.7, label="Rolling 4-week")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(references)))
    i = 0
    for label, lookback in references.items():
        less_df = df.loc[:(df.index.max()-lookback)].iloc[-1]
        ax.hlines(y=less_df['Commercial Crude (Excluding SPR)'],
                  xmin=consider_df.index.min(),
                  xmax=consider_df.index.max(),
                  linewidth=2, alpha=0.3, color=colors[i],
                  label=label)
        i += 1

    ylabels = [ int(l) for l in np.linspace(consider_df['Commercial Crude (Excluding SPR)'].min(), consider_df['Commercial Crude (Excluding SPR)'].max(), 5) ]
    ax.yaxis.set_ticks(ylabels)
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(mdate_fmt)
    if references:
        ax.legend()
    
def populate_dashboard(df, fig, axs):

    price_lookback = 365 * 2
    oil_prices = get_oil_data(days=price_lookback)

    gs = axs[0][0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :])
    
    plot_oil_price(oil_prices, axbig, days=price_lookback)
    
    plot_draw_build_bar(df, axs[1][0], weeks=52)
    plot_draw_build_dist(df, axs[1][1], weeks=52)

    reference_levels = {"2 years ago": dt.timedelta(weeks=52*2),
                        "1 year ago": dt.timedelta(weeks=52),
                        "6 months ago": dt.timedelta(weeks=26),
                        "4 weeks ago": dt.timedelta(weeks=4)}
    
    plot_spr_line(df, axs[2][0], weeks=52*2, references=reference_levels)
    plot_commercial_inventory_line(df, axs[2][1], weeks=52*2, references=reference_levels)


app = Flask(__name__)

@app.route("/")
def dashboard():
    
    inventory_df = add_inventory_features(load_inventory_data())

    fig = Figure(figsize=(14, 10))
    axs = fig.subplots(nrows=3, ncols=2)
    populate_dashboard(inventory_df, fig, axs)
    fig.tight_layout()
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

    


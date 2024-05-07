
import numpy as np
import pandas as pd

import pytz
import datetime as dt

import yfinance as yf

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType

wall_st = pytz.timezone("America/New_York")


def format_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
    # Extract report date from endpoint url
    df['Report Date'] = df['Endpoint'].str.extract(r'(\d{4}_\d{2}_\d{2})')
    # Drop URL, Report Date
    df = df.drop('Endpoint', axis=1)
    df = df.reset_index(drop=True)
    # Set report date as index, convert to DatetimeIndex
    df = df.set_index('Report Date')
    df.index = pd.to_datetime(df.index, format='%Y_%m_%d').tz_localize(wall_st)
    return df

# Inventory Data
stock_summary = pd.read_csv("./reports.csv", index_col=0)
inv = format_inventory_data(stock_summary)
inv['Commercial Crude (Excluding SPR)'] = inv['Crude Oil'] - inv['SPR']
inv['Weekly Commercial Draw/Build'] = inv['Commercial Crude (Excluding SPR)'].diff()

# Market Data
wti = yf.Ticker("CL=F")
hist = wti.history(period="13y")
oil_prices = hist.loc[inv.index.min():inv.index.max() + dt.timedelta(days=7)]

def build_market_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    
    trace = go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                          name='WTI Spot Price')
    return [trace]


def build_stock_bar_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    ## add the bars one at a time
    traces = []

    inv = df['Weekly Commercial Draw/Build'].dropna().copy().to_frame('Change')
    inv['Draw/Build'] = np.where(inv['Change'] >= 0, "Build", "Draw")
    
    color_map = {"Build": 'red', "Draw": 'green'}
    
    for change in ['Draw', 'Build']:
        sub_df = inv[inv['Draw/Build'] == change]
        trace = go.Bar(x=sub_df.index, 
                       y=sub_df['Change'],
                       width=5.184e8, # 6 days in milliseconds
                       name=change,
                       marker_color=color_map[change],
                       opacity=0.6)
        traces.append(trace)
    return traces

def build_crude_line_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    trace = go.Scatter(x=df.index,
                       y=df['Commercial Crude (Excluding SPR)'],
                       name='Commercial Crude Inventory',
                      fill='tozeroy')
    return [trace]

def build_spr_line_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    trace = go.Scatter(x=df.index,
                       y=df['SPR'],
                      mode='lines',
                      name='SPR',
                      fill='tozeroy')
    return [trace]


fig = make_subplots(rows=4, 
                    cols=1, 
                    shared_xaxes=True,
                    subplot_titles=("WTI Spot Price", "Commercial Stock Change",
                                    "Commercial Stock Level", "SPR Stock Level"),
                    vertical_spacing=0.05) 

fig.add_traces(build_market_trace(oil_prices), rows=1, cols=1)
fig.add_traces(build_stock_bar_trace(inv), rows=2, cols=1)
# fig.add_traces(build_stock_box_trace(inv), rows=2, cols=1)
fig.add_traces(build_crude_line_trace(inv), rows=3, cols=1)
fig.add_traces(build_spr_line_trace(inv), rows=4, cols=1)

fig.update_layout(
    title_text="EIA US Crude Inventory Dashboard",
    # autosize=True,
    height=2000,
    xaxis=dict(
        # rangeselector=dict(
        #     buttons=list([
        #         dict(count=1,
        #              label="1m",
        #              step="month",
        #              stepmode="backward"),
        #         dict(count=6,
        #              label="6m",
        #              step="month",
        #              stepmode="backward"),
        #         dict(count=1,
        #              label="YTD",
        #              step="year",
        #              stepmode="todate"),
        #         dict(count=1,
        #              label="1y",
        #              step="year",
        #              stepmode="backward"),
        #         dict(step="all")
        #     ])
        # ),
        rangeslider=dict(
            visible=False
        ),
        type="date"
    )
)


app = Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig, id="fig")
])

@app.callback(
    Output("fig", "figure"),
    Input("fig", "relayoutData"),
)
def scaleYaxis(rng):

    if rng and "xaxis4.range[0]" in rng.keys():

        try:
            x0 = dt.datetime.strptime(rng["xaxis.range[0]"][:10], '%Y-%m-%d')
            x1 = dt.datetime.strptime(rng["xaxis.range[1]"][:10], '%Y-%m-%d')
            x0 = wall_st.localize(x0)
            x1 = wall_st.localize(x1)

            # Price Data
            miny = oil_prices.loc[x0:x1, 'Low'].min()
            maxy = oil_prices.loc[x0:x1, 'High'].max()
            fig['layout']['yaxis']['range'] = [miny, maxy]

            # Crude Change Data
            miny = inv.loc[x0:x1, 'Weekly Commercial Draw/Build'].min()
            maxy = inv.loc[x0:x1, 'Weekly Commercial Draw/Build'].max()
            fig['layout']['yaxis2']['range'] = [miny, maxy]
            
            # Crude Inventory Data
            miny = inv.loc[x0:x1, 'Commercial Crude (Excluding SPR)'].min()
            maxy = inv.loc[x0:x1, 'Commercial Crude (Excluding SPR)'].max()
            fig['layout']['yaxis3']['range'] = [miny, maxy]

            # SPR Inventory Data
            miny = inv.loc[x0:x1, 'SPR'].min()
            maxy = inv.loc[x0:x1, 'SPR'].max()
            fig['layout']['yaxis4']['range'] = [miny, maxy]
            
        except KeyError as e:
            print(e)
            pass

            
        finally:
            
            fig["layout"]["xaxis4"]["range"] = [rng["xaxis4.range[0]"], rng["xaxis4.range[1]"]]

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)


    


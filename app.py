import copy
import pickle

import numpy as np
import pandas as pd

import pytz
import datetime as dt

import yfinance as yf

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType

from xgboost import XGBRegressor

import logging


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

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
logger.info("Loading WPSR data...")
stock_summary = pd.read_csv("./reports.csv", index_col=0)
inv = format_inventory_data(stock_summary)
inv['Commercial Crude'] = inv['Crude Oil'] - inv['SPR']


# Market Data
logger.info("Fetching WTI price data...")
wti = yf.Ticker("CL=F")
hist = wti.history(period="13y")
oil_prices = hist.loc[inv.index.min():inv.index.max() + dt.timedelta(days=7)]


# XGBoost Model
with open('xgb_model.pkl', 'rb') as f:
    model: XGBRegressor = pickle.load(f)
model_features = ['Total Motor Gasoline', 'Fuel Ethanol', 'Kerosene-Type Jet Fuel',
                  'Distillate Fuel Oil', 'Residual Fuel Oil', 'Propane/Propylene',
                  'Other Oils', 'Unfinished Oils', 'Commercial Crude']


# Graph object functions
def build_market_trace(df: pd.DataFrame) -> list[BaseTraceType]:

    trace = go.Scatter(x=df.index,
                       y=df['Close'],
                       mode='lines',
                       name='WTI spot price',
                       fill='tozeroy')
    return [trace]


def build_bar_trace(df: pd.DataFrame,
                    col: str = 'Commercial Crude (Excluding SPR)'
                    ) -> list[BaseTraceType]:

    traces = []

    deltas = df[col].diff().dropna().copy().to_frame('Change')
    deltas['Draw/Build'] = np.where(deltas['Change'] >= 0, "Build", "Draw")

    color_map = {"Build": 'red', "Draw": 'green'}

    for change in ['Draw', 'Build']:
        sub_df = deltas[deltas['Draw/Build'] == change]
        trace = go.Bar(x=sub_df.index,
                       y=sub_df['Change'],
                       opacity=0.6,
                       width=3.456e8,  # 6 days in milliseconds
                       name=change,
                       marker_color=color_map[change])

        traces.append(trace)

    return traces


def build_line_trace(df: pd.DataFrame,
                     col: str = 'Commercial Crude'
                     ) -> list[BaseTraceType]:

    trace = go.Scatter(x=df.index,
                       y=df[col],
                       mode='lines',
                       name=col,
                       fill='tozeroy')

    return [trace]


def build_model_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    last_price_date = oil_prices.index[-1]
    
    lookback = 4
    X = df[model_features].rolling(lookback).mean().to_numpy()
    y = model.predict(X)

    av_cpi = 0.0274
    days_per_year = 365.2425
    today = dt.datetime.now(tz=wall_st)
    years_ago = (today - df.index).days // days_per_year
    inflation_adjs = (1 + av_cpi) ** years_ago.to_numpy()
    true_y = y / inflation_adjs
    print(true_y[-3:])
    
    trace = go.Scatter(x=list(df.index) + [last_price_date],
                    y=list(true_y) + [true_y[-1]],
                    mode='lines',
                    line_shape='hv',
                    name='XGB Prediction')

    return [trace]


def build_figure(col: str = 'Commercial Crude') -> go.Figure:

    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("WTI Spot Price", f"{col} Change",
                                        f"{col} Level"),
                        vertical_spacing=0.05)

    fig.add_traces(build_market_trace(oil_prices), rows=1, cols=1)
    fig.add_traces(build_model_trace(inv), rows=1, cols=1)
    fig.add_traces(build_bar_trace(inv, col), rows=2, cols=1)
    fig.add_traces(build_line_trace(inv, col), rows=3, cols=1)

    fig.update_layout(
        autosize=True,
        height=900,
        showlegend=False,
        yaxis_title="$/barrel",
        yaxis2_title="Million Barrels",
        yaxis3_title="Million Barrels",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=False
            ),
            type="date"
        )
    )

    return fig


def rescale_y_axis(figure, relay):
    """
    """

    new_layout = copy.deepcopy(figure['layout'])

    if "xaxis.range[0]" in relay:

        new_layout['xaxis3']['range'] = [relay['xaxis.range[0]'],
                                         relay['xaxis.range[1]']]

        x0 = dt.datetime.strptime(relay["xaxis.range[0]"][:10], '%Y-%m-%d')
        x1 = dt.datetime.strptime(relay["xaxis.range[1]"][:10], '%Y-%m-%d')
        x0 = wall_st.localize(x0)
        x1 = wall_st.localize(x1)

        print(figure['data'])

        # Price Data
        i0 = np.searchsorted(figure['data'][0]['x'], x0)
        i1 = np.searchsorted(figure['data'][0]['x'], x1)
        miny = min(figure['data'][0]['y'][i0:i1])
        maxy = max(figure['data'][0]['y'][i0:i1])
        new_layout['yaxis']['range'] = [miny*0.98, maxy*1.01]

        # Feature Change Data
        i0 = np.searchsorted(figure['data'][2]['x'], x0)
        i1 = np.searchsorted(figure['data'][2]['x'], x1)
        window_data = figure['data'][2]['y'][i0:i1]
        miny = min(window_data) if window_data.size > 0 else 0

        i0 = np.searchsorted(figure['data'][3]['x'], x0)
        i1 = np.searchsorted(figure['data'][3]['x'], x1)
        window_data = figure['data'][3]['y'][i0:i1]
        maxy = max(window_data) if window_data.size > 0 else 0
        new_layout['yaxis2']['range'] = [miny*0.98, maxy*1.01]

        # Feature Level Data
        i0 = np.searchsorted(figure['data'][4]['x'], x0)
        i1 = np.searchsorted(figure['data'][4]['x'], x1)
        miny = min(figure['data'][4]['y'][i0:i1])
        maxy = max(figure['data'][4]['y'][i0:i1])
        new_layout['yaxis3']['range'] = [miny*0.98, maxy*1.01]

    figure['layout'] = new_layout

    return figure


logger.info("Building dashboard...")
app = Dash()

app.layout = html.Div([
    html.H1(children='EIA Weekly Petroleum Status Report Dashboard',
            style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    dcc.Dropdown(inv.columns, 'Commercial Crude',
                 id='Dropdown', style={'fontFamily': 'sans-serif'}),
    dcc.Graph(figure=build_figure(), id="Graph")
])


@app.callback(
    Output('Graph', 'figure'),
    Input('Dropdown', 'value'),
    Input('Graph', 'relayoutData'),
    State('Graph', 'figure')
)
def plot_new_feature(feature, relay, fig):
    if feature:
        fig = build_figure(feature)

    if relay:
        fig = rescale_y_axis(fig, relay)

    return fig


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)

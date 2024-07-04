import copy
import pickle
import logging

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

from sklearn.pipeline import Pipeline


# Configurations
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

external_stylesheets = [
    {
        'href': (
            'https://fonts.googleapis.com/css2?'
            'family=Lato:wght@400;700&display=swap'
        ),
        'rel': 'stylesheet',
    },
]

wall_st = pytz.timezone('America/New_York')


# Inventory Data
logger.info("Loading WPSR data...")
stock_summary = pd.read_csv("./reports.csv", index_col=0)

def format_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Format inventory DataFrame with report date as index,
    extract Commercial Crude level from Total Crude stocks and SPR
    '''
    # Extract report date from endpoint url
    df['Report Date'] = df['Endpoint'].str.extract(r'(\d{4}_\d{2}_\d{2})')
    
    if df['Report Date'].iloc[-1] is np.nan:
        fmt = '%Y_%m_%d'
        last_valid = dt.datetime.strptime(df['Report Date'].iloc[-2], fmt)
        df['Report Date'].iloc[-1] = (last_valid + dt.timedelta(days=7)).strftime(fmt)

    # Drop URL, Report Date
    df = df.drop('Endpoint', axis=1)
    df = df.reset_index(drop=True)
    # Set report date as index, convert to DatetimeIndex
    df = df.set_index('Report Date')
    df.index = pd.to_datetime(df.index, format='%Y_%m_%d').tz_localize(wall_st)
    # Extract commercial inventory from total crude and SPR
    df['Commercial Crude'] = df['Crude Oil'] - df['SPR']
    print(df.tail(3))
    return df

inv = format_inventory_data(stock_summary)


# Market Data
logger.info('Fetching WTI price data...')
wti = yf.Ticker('CL=F')
hist = wti.history(period='13y')
oil_prices = hist.loc[inv.index.min()
                      :inv.index.max() + dt.timedelta(days=7)]


# XGBoost Model
with open('ridge_model.pkl', 'rb') as f:
    model: Pipeline = pickle.load(f)

model_features = ['Total Motor Gasoline', 'Fuel Ethanol',
                  'Kerosene-Type Jet Fuel', 'Distillate Fuel Oil',
                  'Residual Fuel Oil', 'Propane/Propylene', 'Other Oils',
                  'Unfinished Oils', 'Commercial Crude']


# Graph object functions
def build_market_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    '''Line plot of WTI spot price
    '''
    trace = go.Scatter(x=df.index,
                       y=df['Close'],
                       mode='lines',
                       name='WTI spot price',
                       fill='tozeroy')
    return [trace]


def build_bar_trace(df: pd.DataFrame,
                    col: str = 'Commercial Crude (Excluding SPR)'
                    ) -> list[BaseTraceType]:
    '''Bar plot of weekly inventory change for feature `col`
    '''
    traces = []

    deltas = df[col].diff().dropna().copy().to_frame('Change')
    deltas['Draw/Build'] = np.where(deltas['Change'] >= 0, "Build", "Draw")

    color_map = {'Build': 'red', 'Draw': 'green'}

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
    '''Line plot of inventory level for feature `col`
    '''
    trace = go.Scatter(x=df.index,
                       y=df[col],
                       mode='lines',
                       name=col,
                       fill='tozeroy')

    return [trace]


def build_model_trace(df: pd.DataFrame) -> list[BaseTraceType]:
    '''Step plot of model `fair` price predictions
    '''
    last_price_date = oil_prices.index[-1]

    X = df[model_features].to_numpy()
    y = model.predict(X)

    av_cpi = 0.0274
    days_per_year = 365.2425
    today = dt.datetime.now(tz=wall_st)
    years_ago = (today - df.index).days // days_per_year
    inflation_adjs = (1 + av_cpi) ** years_ago.to_numpy()
    true_y = y / inflation_adjs

    trace = go.Scatter(x=np.append(np.array(df.index), last_price_date),
                       y=np.append(true_y, true_y[-1]),
                       mode='lines',
                       line_shape='hv',
                       name='Model Prediction')

    return [trace]


def build_figure(col: str = 'Commercial Crude') -> go.Figure:
    '''Construct dashboard figures, defaulting to `col` feature
    of inventory DataFrame
    '''
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=('WTI Spot Price', f'{col} Change',
                                        f'{col} Level'),
                        vertical_spacing=0.05)

    fig.add_traces(build_market_trace(oil_prices), rows=1, cols=1)
    fig.add_traces(build_model_trace(inv), rows=1, cols=1)
    fig.add_traces(build_bar_trace(inv, col), rows=2, cols=1)
    fig.add_traces(build_line_trace(inv, col), rows=3, cols=1)

    fig.update_layout(
        autosize=True,
        height=1200,
        showlegend=False,
        yaxis_title='$/barrel',
        yaxis2_title='Million Barrels',
        yaxis3_title='Million Barrels',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='YTD',
                         step='year',
                         stepmode='todate'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=False
            ),
            type="date"
        )
    )

    return fig


def get_window_y_range(x: np.array, y: np.array,
                       x0: dt.datetime, x1: dt.datetime,
                       cushion: float = 0.05) -> tuple[float, float]:
    '''Get trace y-range for xaxis range (x0, x1) with cushioning
    '''
    # Find closest index
    first_index = np.searchsorted(x, x0)
    last_index = np.searchsorted(x, x1)
    # Filter y values
    filtered_y = y[first_index:last_index]
    # Empty edge case
    if filtered_y.size == 0:
        return 0, 0
    # Limit values
    min_y = min(filtered_y)
    max_y = max(filtered_y)
    # Buffer size
    buffer = (max_y - min_y) * cushion
    return min_y - buffer, max_y + buffer


def rescale_y_axis(figure, relay):
    '''Updates yaxis ranges in figure layout according to xaxis relayout
    '''
    expected = {'xaxis.range[0]', 'xaxis.range[1]'}

    if expected.issubset(relay.keys()):

        new_layout = copy.deepcopy(figure['layout'])

        new_x_range = [relay['xaxis.range[0]'],
                       relay['xaxis.range[1]']]

        # Extract xaxis datetime limits
        x0 = dt.datetime.strptime(new_x_range[0][:10], '%Y-%m-%d')
        x1 = dt.datetime.strptime(new_x_range[1][:10], '%Y-%m-%d')
        # Localise datetimes to EST
        x0 = wall_st.localize(x0)
        x1 = wall_st.localize(x1)

        updated_axis = set()

        for trace in figure['data']:
            # Convert axis anchors to layout keys
            xaxis = trace['xaxis'].replace('x', 'xaxis')
            yaxis = trace['yaxis'].replace('y', 'yaxis')
            # Trace data
            x = trace['x']
            y = trace['y']
            # New y range
            this_y_range = get_window_y_range(x, y, x0, x1)

            if yaxis in updated_axis:  # In case of multiple traces
                curr_y_range = new_layout[yaxis]['range']
                new_y_range = [min(this_y_range[0], curr_y_range[0]),
                               max(this_y_range[1], curr_y_range[1])]
            else:
                new_y_range = list(this_y_range)

            # Update layout range
            new_layout[xaxis]['range'] = new_x_range
            new_layout[yaxis]['range'] = new_y_range
            # Remember visited axis
            updated_axis.add(yaxis)

        # Replace old layout
        figure['layout'] = new_layout

    return figure


logger.info('Building dashboard...')
app = Dash(title='WTI Inventory', external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H1(children='EIA Weekly Petroleum Status Report Dashboard',
                className='header-title'),
            html.P(
                children=(
                    'Analysis of EIA Petroleum Inventory and WTI Spot Price'
                ),
                className='header-description',
            ),
        ],
        className='header'),
    dcc.Dropdown(inv.columns, 'Commercial Crude',
                 id='Dropdown', className='menu'),
    dcc.Graph(figure=build_figure(), id='Graph', className='card')
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

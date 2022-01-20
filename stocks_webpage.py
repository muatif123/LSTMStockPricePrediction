import imp
from statistics import mode
from tkinter import X
from turtle import title
import dash
import dash_core_components as dcc
import dash_html_components as dhc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range = (0, 1))

df_nse = pd.read_csv("./nse_tata.csv")
df_nse['Date'] = pd.to_datetime(df_nse.Date, format = '%Y-%m-%d')
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending = True, axis = 0)
new_df = pd.DataFrame(index = range(0, len(df_nse)), columns = ['Date', 'Close'])

for i in range(0, len(data)):
    new_df['Date'][i] = data['Date'][i]
    new_df['Close'][i] = data['Close'][i]

new_df.index = new_df.Date
new_df.drop('Date', axis = 1, inplace = True)

df_set = new_df.values

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(df_set)

train_df = df_set[0:987, :]
valid_df = df_set[987:, :]

X_train, Y_train = [], []

for i in range(60, len(train_df)):
    X_train.append(scaled_data[i-60:i, 0])
    Y_train.append(scaled_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = load_model('saved_model.h5')

inputs = new_df[len(new_df) - len(valid_df)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train_df = new_df[:987]
valid_df = new_df[987:]
valid_df['Predictions'] = closing_price

######
df = pd.read_csv("./stock_data.csv")

app.layout = dhc.Div([
    dhc.H1("Stock Price Analysis Dashboard", style = {"textAlign": "center"}),
    dcc.Tabs(id = 'tabs', children = [
        dcc.Tab(label = 'TATA Global Stock', children = [
            dhc.Div([
                dhc.H2("Actual Closing Price", style = {"textAlign": "center"}),
                dcc.Graph(
                    id = 'Actual Data',
                    figure = {
                        "data": [
                            go.Scatter(
                                x = train_df.index,
                                y = valid_df['Close'],
                                mode = 'markers'
                            )
                        ],
                        "layout": go.Layout(
                            title = 'Scatter Plot',
                            xaxis = {'title': 'Date'},
                            yaxis = {'title': 'Closing Rate'}
                        )
                    }
                ),
                dhc.H2("LSTM Predictied Closing Price", style = {'textAlign': 'center'}),
                dcc.Graph(
                    id = "Predicted Data",
                    figure = {
                        "data": [
                            go.Scatter(
                                x = valid_df.index,
                                y = valid_df['Predictions'],
                                mode = 'markers'
                            )
                        ],
                        "layout": go.Layout(
                            title = 'Scatter Plot',
                            xaxis = {'title': 'Date'},
                            yaxis = {'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label = 'Facebook Stock Data', children = [
            dhc.Div([
                dhc.H1(f"Facebook Stock High vs Low", style = {'textAlign': 'center'}),
                dcc.Dropdown(id = 'my-dropdown', options = [{
                    'label': 'Tesla', 'value': 'TSLA'
                },
                {
                    'label': 'Apple', 'value': 'AAPL'
                },
                {
                    'label': 'Facebook', 'value': 'FB'
                },
                {
                    'label': 'Microsoft', 'value': 'MSFT'
                }],
                multi = True, value = ['FB'], style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '60%'}),
                dcc.Graph(id = 'highlow'),
                dhc.H1('Facebook Market Volume', style = {'textAlign': 'center'}),
                dcc.Dropdown(id = 'my-dropdown2', options = [{
                    'label': 'Tesla', 'value': 'TSLA'
                },
                {
                    'label': 'Apple', 'value': 'AAPL'
                },
                {
                    'label': 'Facebook', 'value': 'FB'
                },
                {
                    'label': 'Microsoft', 'value': 'MSFT'
                }],
                multi = True, value = ['FB'], style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '60%'}),
                dcc.Graph(id = 'volume')
            ],
            className = 'container'),
        ])
    ])
])

@app.callback(Output('highlow', 'figure'),[Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": 'Tesla', "AAPL": 'Apple', "FB": 'Facebook', "MSFT": 'Microsoft',}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x = df[df['Stock'] == stock]['Date'],
                       y = df[df['Stock'] == stock]['High'],
                       mode = 'lines', opacity = 0.7,
                       name = f'High {dropdown[stock]}',
                       textposition = 'bottom center')
        )
        trace2.append(
            go.Scatter(x = df[df['Stock'] == stock]['Date'],
                       y = df[df['Stock'] == stock]['Low'],
                       mode = 'lines', opacity = 0.6,
                       name = f'Low {dropdown[stock]}',
                       textposition = 'bottom center')
        )
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(
                      colorway = ["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
                      height = 600, title = f"High and Low Prices for {','.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                      xaxis = {"title": 'Date',
                               "rangeselector": {'buttons': list([{
                                   'count': 1,
                                   'label': '1M',
                                   'step': 'month',
                                   'stepmode': 'backward'
                               },
                               {
                                   'count': 6,
                                   'label': '6M',
                                   'step': 'month',
                                   'stepmode': 'backward'
                               },
                               {
                                   'step': 'all'
                               }])},
                               'rangeslider': {'visible': True}, 'type': 'date'},
                      yaxis = {"title": "Price (USD)"}
                  )}
    return figure

@app.callback(Output('volume', 'figure'),[Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": 'Tesla', "AAPL": 'Apple', "FB": 'Facebook', "MSFT": 'Microsoft',}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x = df[df['Stock'] == stock]['Date'],
                       y = df[df['Stock'] == stock]['Volume'],
                       mode = 'lines', opacity = 0.7,
                       name = f'Volume {dropdown[stock]}',
                       textposition = 'bottom center')
        )
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(
                      colorway = ["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
                      height = 600, title = f"Market Volume for {','.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                      xaxis = {"title": 'Date',
                               "rangeselector": {'buttons': list([{
                                   'count': 1,
                                   'label': '1M',
                                   'step': 'month',
                                   'stepmode': 'backward'
                               },
                               {
                                   'count': 6,
                                   'label': '6M',
                                   'step': 'month',
                                   'stepmode': 'backward'
                               },
                               {
                                   'step': 'all'
                               }])},
                               'rangeslider': {'visible': True}, 'type': 'date'},
                      yaxis = {"title": "Price (USD)"}
                  )}
    return figure

if __name__=='__main__':
    app.run_server(debug = True)
                
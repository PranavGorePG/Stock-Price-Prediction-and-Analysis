import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, jsonify


import io
import base64
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Load the Keras model
model = load_model('keras_model.h5')

# Create an instance of Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    user_input = request.form['ticker']
    user_input = str(user_input)

    # Get the stock data
    start = '2010-01-01'
    end = '2023-2-28'
    yfin.pdr_override()
    df = pdr.get_data_yahoo(user_input, start, end)

    fig1,ax1 = plt.subplots(figsize = (12,6))
    ax1.plot(df.Close)
    img = io.BytesIO()
    fig1.savefig(img, format='png')
    img.seek(0)

    # encode the PNG image as base64 string
    plot_url1 = base64.b64encode(img.getvalue()).decode()

    ma100 = df.Close.rolling(100).mean()

    fig2, ax2 = plt.subplots(figsize = (12,6))
    ax2.plot(ma100)
    ax2.plot(df.Close)
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    img.seek(0)

    # encode the PNG image as base64 string
    plot_url2 = base64.b64encode(img2.getvalue()).decode()



    #Splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)



    #Load the Model
    model = load_model('keras_model.h5')

    #Testing part

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index = True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler =scaler.scale_


    scale_factor = 1/scaler[0]
    y_predicted = y_predicted *scale_factor
    y_test = y_test * scale_factor

    fig3, ax3 = plt.subplots(figsize = (12,6))
    ax3.plot(y_test, 'b', label = 'Orignal Price')
    ax3.plot(y_predicted, 'r', label = 'Predicted Price')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Price')
    ax3.legend(loc='upper left')

    img3 = io.BytesIO()
    fig3.savefig(img3, format='png')
    img.seek(0)

    # encode the PNG image as base64 string
    plot_url3 = base64.b64encode(img3.getvalue()).decode()



    return render_template('predict.html',plot_url1 = plot_url1,plot_url2 = plot_url2,
                            plot_url3 = plot_url3)

@app.route('/template')
def template():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

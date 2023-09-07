import tensorflow as tf
import numpy as np
from flask import Flask,redirect,url_for,render_template,request
import uvicorn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
import webbrowser
#Fetch historical stock data
app = Flask(__name__)
@app.route('/update_content', methods=['POST'])
def update_content():
    new_content = request.form['new_content']
    ticker = new_content
    start_date = "2022-05-16"
    end_date = "2023-06-07"
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    dataset = data.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    test_size = int(len(scaled_data))
    test_data = scaled_data
    inputs = data[len(data) - len(test_data):].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    window_size = 60
    X_test = []
    for i in range(window_size, len(inputs)):
        X_test.append(inputs[i - window_size:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    app=Flask(__name__)
    match ticker:
        case "GOOGL":
            MODEL = tf.keras.models.load_model(f"D:/Users/sakthi/Documents/VSBEC/PROJECT_SOURCE/Stock/Models/model_GOOGL.h5" )
        case "MSFT":
            MODEL = tf.keras.models.load_model(f"D:/Users/sakthi/Documents/VSBEC/PROJECT_SOURCE/Stock/Models/model_MSFT.h5" )
        case "TSLA":
            MODEL = tf.keras.models.load_model(f"D:/Users/sakthi/Documents/VSBEC/PROJECT_SOURCE/Stock/Models/model_TSLA.h5" )
        case "AAPL":
            MODEL = tf.keras.models.load_model(f"D:/Users/sakthi/Documents/VSBEC/PROJECT_SOURCE/Stock/Models/model_AAPL.h5" )
        case "BABA":
            MODEL = tf.keras.models.load_model(f"D:/Users/sakthi/Documents/VSBEC/PROJECT_SOURCE/Stock/Models/model_BABA.h5" )
        case "AMZN":
            MODEL = tf.keras.models.load_model(f"D:/Users/sakthi/Documents/VSBEC/PROJECT_SOURCE/Stock/Models/model_AMZN.h5" )
    
    predictions = MODEL.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.tolist()
    msg =  predictions[-1]
    return msg

@app.route('/', methods=['GET', 'POST'])
def redirect_to_html():
    if request.method == 'POST':
        selected_option = request.form.get('option')
        return f"You selected: {selected_option}"
    return  render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

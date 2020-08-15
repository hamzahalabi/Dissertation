import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import plotly.express as px
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
import tensorflow as tf
from keras import optimizers
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from project import StockPrediction
import math
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

class LSTM_Models(StockPrediction):

    def lstm_model_results(self):
        df_google = self.create_raw_dataset('GOOG')
        df_apple = self.create_raw_dataset('AAPL')
        df_google = self.add_all_indicators(df_google,'Close','High','Low','Volume')
        df_apple = self.add_all_indicators(df_apple,'Close','High','Low','Volume')

        df_google['target'] = df_google['Adj Close']

        for x in range(len(df_google['Adj Close'])-1):
            if df_google['Adj Close'][x] < df_google['Adj Close'][x+1]:
                df_google['target'][x]=1
            else:
                df_google['target'][x]=-1

        df_apple['target'] = df_apple['Adj Close']

        for x in range(len(df_apple['Adj Close'])-1):
            if df_apple['Adj Close'][x] < df_apple['Adj Close'][x+1]:
                df_apple['target'][x]=1
            else:
                df_apple['target'][x]=-1

        df_apple=df_apple[99:-1].reset_index().drop('index',axis=1)
        df_google=df_google[99:-1].reset_index().drop('index',axis=1)

        X = np.array(df_apple[['Adj Close','macd', 'macd_signal', 'stoch', 'stoch_signal', 'roc', 'cci',
           'adi','rsi','wr']])
        y = np.array(pd.get_dummies(df_apple['target'])[1.0])

        scaler = MinMaxScaler(feature_range=(-1,1))
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_val = pd.get_dummies(df_google['target'])[1.0]
        y_val = y_val[3072:]

        model = load_model('aaplmodelcont.h5')
        acc = accuracy_score(y_val,model.predict_classes(X_test))

        print(f'LSTM accuracy for Google baseline approach: {100*acc:.2f}%')

    def create_lstm_model(self):
        df_google = self.create_raw_dataset('GOOG')
        df_apple = self.create_raw_dataset('AAPL')
        df_google = self.add_all_indicators(df_google,'Close','High','Low','Volume')
        df_apple = self.add_all_indicators(df_apple,'Close','High','Low','Volume')

        df_google['target'] = df_google['Adj Close']

        for x in range(len(df_google['Adj Close'])-1):
            if df_google['Adj Close'][x] < df_google['Adj Close'][x+1]:
                df_google['target'][x]=1
            else:
                df_google['target'][x]=-1

        df_apple['target'] = df_apple['Adj Close']

        for x in range(len(df_apple['Adj Close'])-1):
            if df_apple['Adj Close'][x] < df_apple['Adj Close'][x+1]:
                df_apple['target'][x]=1
            else:
                df_apple['target'][x]=-1

        df_apple=df_apple[99:-1].reset_index().drop('index',axis=1)
        df_google=df_google[99:-1].reset_index().drop('index',axis=1)

        X = np.array(df_apple[['Adj Close','macd', 'macd_signal', 'stoch', 'stoch_signal', 'roc', 'cci',
           'adi','rsi','wr']])
        y = np.array(pd.get_dummies(df_apple['target'])[1.0])

        scaler = MinMaxScaler(feature_range=(-1,1))
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_val = pd.get_dummies(df_google['target'])[1.0]
        y_val = y_val[3072:]

        ad = optimizers.Adam(learning_rate=0.0001)
        model = Sequential()
        model.add(LSTM(16,return_sequences=False,input_shape=(1,9)))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=ad, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=32,validation_data=(X_test,y_test),shuffle=False)
        

    


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
import math
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

class StockPrediction():

    # Function to create the raw datasets for Apple and Google each.
    def create_raw_dataset(self,stock):
        start = dt.datetime(2004,8,20)
        end = dt.datetime(2020,4,14)
        df = web.DataReader(stock, 'yahoo',start,end)
        df.reset_index(inplace=True)
        return df

    # Williams R% indicator
    def wr(self,df,n):
        hh = df['High'].rolling(n).max()
        ll = df['Low'].rolling(n).min()
        cp = df['Adj Close']
        wr = -100* (hh-cp)/(hh-ll)
        return(pd.Series(wr))

    # Calculates Exponential Moving Average and returns a Series
    def ema(self,series,n):
        ema = series.ewm(span=n,min_periods=n,adjust=False).mean()
        return(pd.Series(ema))

    # Calculates Moving Average Convergence/Divergence and returns a Series
    def macd(self,series,fast,slow):
        fastema = self.ema(series,fast)
        slowema = self.ema(series,slow)
        macd = fastema - slowema
        return (pd.Series(macd))

    #MACD signal indicator
    def macd_signal(self,macd,n):
        macd_signal = self.ema(macd,n)
        return (pd.Series(macd_signal))

    # Simple Moving Average indicator
    def sma(self,series,n):
        sma = series.rolling(n).mean()
        return (pd.Series(sma))

    # Relative Strength Index indicator
    def rsi(self,close,n):
        change = close.diff()
        gain = change.where(change>0,0)
        loss = -change.where(change<0,0)
        avggain = gain.ewm(alpha=1/n, min_periods=0, adjust=False).mean()
        avgloss = loss.ewm(alpha=1/n, min_periods=0, adjust=False).mean()
        rsf = avggain/avgloss
        rsi = 100-(100/(1+rsf))
        return (pd.Series(rsi))

    # Stochastic Oscillator %K indicator
    def stoch(self,df,n):
        hh = df['High'].rolling(n).max()
        ll = df['Low'].rolling(n).min()
        cp = df['Adj Close']
        stoch = 100*(cp-ll)/(hh-ll)
        return(pd.Series(stoch))

    # Stochastic Signal %D indicator
    def stoch_signal(self,stoch,n):
        stoch_signal = self.sma(stoch,n)
        return (pd.Series(stoch_signal))

    # Rate of Change indicator
    def roc(self,close,n):
        change = close.diff(n)
        perchange = 100* (change/close.shift(n))
        return (pd.Series(perchange))

    # Commodity Channel Index indicator
    def cci(self,df,n,c=.015):
        def dev(x):
            return np.mean(np.abs(x-np.mean(x)))
        h = df['High']
        l = df['Low']
        cp = df['Adj Close']
        tp = (h+l+cp)/3
        tpma = self.sma(tp,n)
        md = np.abs(self.sma((tpma-tp),n).sum())
        cci = (tp-tpma) / (c*tp.rolling(n).apply(dev,True))
        return (pd.Series(cci))

    #Accumulation/Distrubution Index indicator
    def adi(self,df):
        h = df['High']
        l = df['Low']
        cp = df['Adj Close']
        vol = df['Volume']
        cmfv = vol* (((cp-l) - (h-cp)) / (h-l) )
        adi = cmfv.cumsum()
        return (pd.Series(adi))

    # Single function to add all technical indicators at once
    def add_all_indicators(self,df,close,high,low,volume):
        df['ema_12'] = self.ema(df[close],12)
        df['ema_26'] = self.ema(df[close],26)
        df['sma_100'] = self.sma(df[close],100)
        df['macd'] = self.macd(df[close],12,26)
        df['macd_signal'] = self.macd_signal(df['macd'],9)
        df['stoch'] = self.stoch(df,14)
        df['stoch_signal'] = self.stoch_signal(df['stoch'],3)
        df['rsi'] = self.rsi(df[close],14)
        df['wr'] = self.wr(df,14)
        df['roc'] = self.roc(df[close],12)
        df['cci'] = self.cci(df,20)
        df['adi'] = self.adi(df)
        return df

    # MACD first opinion column
    def macd_op(self,df):
        col = []
        for x in range(len(df['macd'])):
            if df['macd'][x] > df['macd_signal'][x]:
                col.append(1)
            else:
                col.append(-1)
        return (pd.Series(col))

    #MACD second opinion column
    def macd_op2(self,df):
        col = []
        for x in range(1,len(df['macd'])):
            if df['macd'][x] > df['macd'][x-1]:
                col.append(1)
            else:
                col.append(-1)
        return (pd.Series(col))

    # MACD difference rate of change
    def macd_diff_roc(self,df):
        xl=[]
        y=[]
        z=[]
        for x in range(len(df['macd'])):
            xl.append(df['macd'][x] - df['macd_signal'][x])
        for x in range(len(df['macd'])):
            y.append(xl[x] - xl[x-1])
        for x in range(1,len(df['macd'])):
            z.append(y[x] - y[x-1])
        return (pd.Series(z))

    # MACD third opinion column 
    def macd_op3(self,df):
        diff = self.macd_diff_roc(df)
        col = []
        for x in range(1,len(diff)):
            if diff[x] > diff[x-1]:
                col.append(1)
            else:
                col.append(-1)
        return (pd.Series(col))

    # Stochastic Oscillators opinion index
    def stoch_op(self,df):
        col = []
        for x in range(len(df['stoch'])):
            if df['stoch'][x] > df['stoch_signal'][x]:
                col.append(1)
            else:
                col.append(-1)
        return (pd.Series(col))

    # Relative Strength Index opinion column
    def rsi_op(self,df):
        col = []
        for x in range(1,len(df['rsi'])):
            if df['rsi'][x] > 80:
                col.append(-1)
            elif df['rsi'][x] < 20:
                col.append(1)
            else:
                if df['rsi'][x] > df['rsi'][x-1]:
                    col.append(1)
                else:
                    col.append(-1)
        return (pd.Series(col))

    # Williams R% opinion column
    def wr_op(self,df):
        col=[]
        for x in range(1,len(df['wr'])):
            if df['wr'][x] > -20:
                col.append(-1)
            elif df['wr'][x] < -80:
                col.append(1)
            else:
                if df['wr'][x] > df['wr'][x-1]:
                    col.append(1)
                else:
                    col.append(-1)
        return (pd.Series(col))

    # Commodity Channel Index opinion column
    def cci_op(self,df):
        col=[]
        for x in range(1,len(df['cci'])):
            if df['cci'][x] > 200:
                col.append(-1)
            elif df['cci'][x] < -200:
                col.append(1)
            elif df['cci'][x] > 100:
                if df['cci'][x] > df['cci'][x-1]:
                    col.append(1)
                else:
                    col.append(-1)
            elif df['cci'][x] < -100:
                if df['cci'][x] > df['cci'][x-1]:
                    col.append(-1)
                else:
                    col.append(1)
            else:
                if df['cci'][x] > df['cci'][x-1]:
                    col.append(1)
                else:
                    col.append(-1)
        return(pd.Series(col))

    # Rate of Change opinion column
    def roc_op(self,df):
        col=[]
        for x in range(len(df['roc'])):
            if df['roc'][x] > 0:
                col.append(1)
            else:
                col.append(-1)
        return(pd.Series(col))

    #Accumulation/Distribution Index opinion column
    def adi_op(self,df):
        col=[]
        for x in range(1,len(df['adi'])):
            if df['adi'][x] > df['adi'][x-1] and df['Adj Close'][x] > df['Adj Close'][x-1]:
                col.append(1)
            elif df['adi'][x] < df['adi'][x-1] and df['Adj Close'][x] < df['Adj Close'][x-1]:
                col.append(-1)
            elif df['adi'][x] > df['adi'][x-1] and df['Adj Close'][x] < df['Adj Close'][x-1]:
                col.append(1)
            else:
                col.append(-1)    
        return(pd.Series(col))

    #Single function to add all technical indicator opinions together
    def add_opinions(self,df):
        df['macd_op']= self.macd_op(df)
        df['macd_op2']= self.macd_op2(df)
        df['macd_op3']= self.macd_op3(df)
        df['stoch_op']= self.stoch_op(df)
        df['roc_op']= self.roc_op(df)
        df['rsi_op']= self.rsi_op(df)
        df['wr_op']= self.wr_op(df)
        df['adi_op']= self.adi_op(df)
        df['cci_op']= self.cci_op(df)
        df['cci_op']=df['cci_op'].shift(1)
        df['wr_op']=df['wr_op'].shift(1)
        df['adi_op']=df['adi_op'].shift(1)
        df['rsi_op']=df['rsi_op'].shift(1)
        df['macd_op2']=df['macd_op2'].shift(1)
        df['macd_op3']=df['macd_op3'].shift(2)
        df = df[2:].reset_index().drop('index',axis=1)
        return(df)    

    # A comprehensive trading simulation algorithm with print statements along the way to show each trade made, how much it was bought for,
    # how much it was sold for and whether it was a losing or winning trade, along with cummulation net profit after each trade.
    # the tradetest function takes the predictions of models and the closing prices at the time of the testing set as arguments
    def tradetest(self,prediction,closeprice,capital=100000):
        switch = 0
        shares = 0
        cumprofit=0
        bp = 0
        sp = 0
        wincount=0
        losecount=0
        moneybefore = capital
        for x in range(len(prediction)):
            
            if prediction[x] == 1  and capital>closeprice[x] and switch!=1:
                shares = capital/closeprice[x]
                switch = 1
                bp = closeprice[x]
                print (f'Buy @{closeprice[x]} Number of shares = {shares} at index {x} \n')
                
            elif prediction[x] == -1 and switch==1:
                capital = shares*closeprice[x]
                cumprofit = capital-moneybefore
                shares = 0
                switch = 0 
                sp = closeprice[x]
                print (f'Sell @{closeprice[x]} Money after sale = {capital} at index {x}')
                print (f'Cummulative Profit = {cumprofit}')
                
                
                if sp > bp:
                    print('Winning Trade \n')
                    wincount+=1
                else:
                    print('Losing Trade \n')
                    losecount+=1
            else:
                continue
        print(f'Total Winning Trades = {wincount} , losing trades = {losecount}, Profitability = {wincount*100/(wincount+losecount):.2f}%')
        print(f'Money Before: ${moneybefore} (March 24th, 2017)')
        print(f'Money After: ${capital:.2f} (April 13th, 2020)')
        print(f'Net Profit : ${cumprofit:.2f}')
        
    # The same trading simulation algorithm but only returns the netprofit at the end without prints within, for use within functions
    def tradetestreturn(self,prediction,closeprice,capital=100000):
        switch = 0
        shares = 0
        cumprofit=0
        bp = 0
        sp = 0
        wincount=0
        losecount=0
        moneybefore = capital
        for x in range(len(prediction)):
            
            if prediction[x] == 1  and capital>closeprice[x] and switch!=1:
                shares = capital/closeprice[x]
                switch = 1
                bp = closeprice[x]
                
            elif prediction[x] == -1 and switch==1:
                capital = shares*closeprice[x]
                cumprofit = capital-moneybefore
                shares = 0
                switch = 0 
                sp = closeprice[x]
                if sp > bp:
                    wincount+=1
                else:
                    losecount+=1
            else:
                continue
        return(cumprofit)

    # Google Prediction Models With The Baseline Approach. SVM,RF and KNN along with a trading simulation for each
    def google_cont_models(self):
        df = self.create_raw_dataset('GOOG')
        df = self.add_adill_incators(df,'Close','High','Low','Volume')
        df['target'] = df['Adj Close']

        for x in range(len(df['Adj Close'])-1):
            if df['Adj Close'][x] < df['Adj Close'][x+1]:
                df['target'][x]=1
            else:
                df['target'][x]=-1

        df=df[99:-1].reset_index().drop('index',axis=1)

        X = df[['Adj Close', 'macd', 'macd_signal', 'stoch', 'stoch_signal', 'roc', 'cci',
           'adi','rsi','wr']]
        y = df['target']

        scaler = MinMaxScaler(feature_range=(-1,1))
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        tscv = TimeSeriesSplit(n_splits=5)

        svm_param_grid = {'C': [2**x for x in range(-5,5)], 'gamma': [2**x for x in range(-7,1)], 'kernel': ['rbf']} 
        svm_grid = GridSearchCV(SVC(),svm_param_grid,verbose=1,cv=tscv,n_jobs=-1).fit(X_train,y_train)
        svm_model = SVC(kernel='rbf',C=0.5,gamma=1).fit(X_train,y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test,svm_pred)

        rf_param_grid = {'bootstrap': [False], 'max_depth': [None],'max_features': [None],
        'min_samples_leaf': [200,250,300],'min_samples_split': [2,4,8,10],
        'n_estimators': [100]}
        rfgrid = GridSearchCV(RandomForestClassifier(),param_grid=rf_param_grid,cv=tscv,scoring='accuracy',n_jobs=-1,verbose=1).fit(X_train,y_train)
        rf_model = RandomForestClassifier(bootstrap=False,n_estimators=100,min_samples_leaf=200,min_samples_split=20).fit(X_train,y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test,rf_pred)

        knn_param_grid = {'n_neighbors':[x for x in range(100)]}
        knngrid = GridSearchCV(KNeighborsClassifier(),param_grid=knn_param_grid,cv=tscv,scoring='accuracy',verbose=1,n_jobs=-1).fit(X_train,y_train)
        knn_model = KNeighborsClassifier(n_neighbors=31).fit(X_train,y_train)
        knn_pred = knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test,knn_pred)

        print('Prediction Accuracy of Google stock with the baseline approach: \n')
        print(f'SVM Model Accuracy : {100*svm_acc:.2f}%')
        print(f'RF Model Accuracy : {100*rf_acc:.2f}%')
        print(f'KNN Model Accuracy : {100*knn_acc:.2f}%')
        trade_svm = self.tradetestreturn(svm_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_rf = self.tradetestreturn(rf_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_knn = self.tradetestreturn(knn_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])

        print(f'SVM trade test Net Profit: ${trade_svm:.2f}')
        print(f'RF trade test Net Profit: ${trade_rf:.2f}')
        print(f'KNN trade test Net Profit: ${trade_knn:.2f}')
        print(len(X_train))


    # Apple Prediction Models With The Baseline Approach. SVM,RF and KNN along with a trading simulation for each
    def apple_cont_models(self):
        df = self.create_raw_dataset('AAPL')
        df = self.add_all_indicators(df,'Close','High','Low','Volume')
        df['target'] = df['Adj Close']

        for x in range(len(df['Adj Close'])-1):
            if df['Adj Close'][x] < df['Adj Close'][x+1]:
                df['target'][x]=1
            else:
                df['target'][x]=-1

        df=df[99:-1].reset_index().drop('index',axis=1)

        X = df[['macd', 'macd_signal', 'stoch', 'stoch_signal', 'roc', 'cci',
           'adi','rsi','wr']]
        y = df['target']

        scaler = MinMaxScaler(feature_range=(-1,1))
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        tscv = TimeSeriesSplit(n_splits=5)

        svm_param_grid = {'C': [2**x for x in range(-5,5)], 'gamma': [2**x for x in range(-7,1)], 'kernel': ['rbf']} 
        svm_grid = GridSearchCV(SVC(),svm_param_grid,verbose=1,cv=tscv,n_jobs=-1).fit(X_train,y_train)
        svm_model = SVC(kernel='rbf',C=128,gamma=1).fit(X_train,y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test,svm_pred)

        rf_param_grid = {'bootstrap': [False], 'max_depth': [None],'max_features': [None],
        'min_samples_leaf': [200,250,300],'min_samples_split': [2,4,8,10],
        'n_estimators': [100]}
        rfgrid = GridSearchCV(RandomForestClassifier(),param_grid=rf_param_grid,cv=tscv,scoring='accuracy',n_jobs=-1,verbose=1).fit(X_train,y_train)
        rf_model = RandomForestClassifier(bootstrap=False,n_estimators=200,min_samples_leaf=200,min_samples_split=10).fit(X_train,y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test,rf_pred)

        knn_param_grid = {'n_neighbors':[x for x in range(100)]}
        knngrid = GridSearchCV(KNeighborsClassifier(),param_grid=knn_param_grid,cv=tscv,scoring='accuracy',verbose=1,n_jobs=-1).fit(X_train,y_train)
        knn_model = KNeighborsClassifier(n_neighbors=70).fit(X_train,y_train)
        knn_pred = knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test,knn_pred)

        print('Prediction Accuracy of Apple stock with the baseline approach: \n')
        print(f'SVM Model Accuracy : {100*svm_acc:.2f}%')
        print(f'RF Model Accuracy : {100*rf_acc:.2f}%')
        print(f'KNN Model Accuracy : {100*knn_acc:.2f}%')
        trade_svm = self.tradetestreturn(svm_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_rf = self.tradetestreturn(rf_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_knn = self.tradetestreturn(knn_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])

        print(f'SVM trade test Net Profit: ${trade_svm:.2f}')
        print(f'RF trade test Net Profit: ${trade_rf:.2f}')
        print(f'KNN trade test Net Profit: ${trade_knn:.2f}')


    # Apple Prediction Models With The Proposed Approach. SVM,RF and KNN along with a trading simulation for each
    def apple_op_models(self):
        df = self.create_raw_dataset('AAPL')
        df = self.add_all_indicators(df,'Close','High','Low','Volume')
        df - self.add_opinions(df)
        df['target'] = df['Adj Close']

        for x in range(len(df['Adj Close'])-1):
            if df['Adj Close'][x] < df['Adj Close'][x+1]:
                df['target'][x]=1
            else:
                df['target'][x]=-1

        df=df[99:-1].reset_index().drop('index',axis=1)

        X = df[['macd_op','macd_op2','macd_op3','roc_op','stoch_op',
             'rsi_op', 'wr_op', 'cci_op','adi_op']]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        tscv = TimeSeriesSplit(n_splits=5)

        svm_param_grid = {'C': [2**x for x in range(-5,5)], 'gamma': [2**x for x in range(-7,1)], 'kernel': ['rbf']} 
        svm_grid = GridSearchCV(SVC(),svm_param_grid,verbose=1,cv=tscv,n_jobs=-1).fit(X_train,y_train)
        svm_model = SVC(kernel='rbf',C=2,gamma=.085).fit(X_train,y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test,svm_pred)

        rf_param_grid = {'bootstrap': [False], 'max_depth': [None],'max_features': [None],
        'min_samples_leaf': [200,250,300],'min_samples_split': [2,4,8,10],
        'n_estimators': [100]}
        rfgrid = GridSearchCV(RandomForestClassifier(),param_grid=rf_param_grid,cv=tscv,scoring='accuracy',n_jobs=-1,verbose=1).fit(X_train,y_train)
        rf_model = RandomForestClassifier(bootstrap=False,n_estimators=100,min_samples_leaf=8,min_samples_split=8).fit(X_train,y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test,rf_pred)

        knn_param_grid = {'n_neighbors':[x for x in range(100)]}
        knngrid = GridSearchCV(KNeighborsClassifier(),param_grid=knn_param_grid,cv=tscv,scoring='accuracy',verbose=1,n_jobs=-1).fit(X_train,y_train)
        knn_model = KNeighborsClassifier(n_neighbors=121).fit(X_train,y_train)
        knn_pred = knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test,knn_pred)

        print('Prediction Accuracy of Apple stock with the opinions approach: \n')
        print(f'SVM Model Accuracy : {100*svm_acc:.2f}%')
        print(f'RF Model Accuracy : {100*rf_acc:.2f}%')
        print(f'KNN Model Accuracy : {100*knn_acc:.2f}%')
        trade_svm = self.tradetestreturn(svm_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_rf = self.tradetestreturn(rf_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_knn = self.tradetestreturn(knn_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])

        print(f'SVM trade test Net Profit: ${trade_svm:.2f}')
        print(f'RF trade test Net Profit: ${trade_rf:.2f}')
        print(f'KNN trade test Net Profit: ${trade_knn:.2f}')

    # Google Prediction Models With The Proposed Approach. SVM,RF and KNN
    def google_op_models(self):
        df = self.create_raw_dataset('GOOG')
        df = self.add_all_indicators(df,'Close','High','Low','Volume')
        df - self.add_opinions(df)
        df['target'] = df['Adj Close']

        for x in range(len(df['Adj Close'])-1):
            if df['Adj Close'][x] < df['Adj Close'][x+1]:
                df['target'][x]=1
            else:
                df['target'][x]=-1

        df=df[99:-1].reset_index().drop('index',axis=1)

        X = df[['macd_op','macd_op2','macd_op3','roc_op','stoch_op',
             'rsi_op', 'wr_op', 'cci_op','adi_op']]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        tscv = TimeSeriesSplit(n_splits=5)

        svm_param_grid = {'C': [2**x for x in range(-5,5)], 'gamma': [2**x for x in range(-7,1)], 'kernel': ['rbf']} 
        svm_grid = GridSearchCV(SVC(),svm_param_grid,verbose=1,cv=tscv,n_jobs=-1).fit(X_train,y_train)
        svm_model = SVC(kernel='rbf',C=1,gamma=1).fit(X_train,y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test,svm_pred)

        rf_param_grid = {'bootstrap': [False], 'max_depth': [None],'max_features': [None],
        'min_samples_leaf': [200,250,300],'min_samples_split': [2,4,8,10],
        'n_estimators': [100]}
        rfgrid = GridSearchCV(RandomForestClassifier(),param_grid=rf_param_grid,cv=tscv,scoring='accuracy',n_jobs=-1,verbose=1).fit(X_train,y_train)
        rf_model = RandomForestClassifier(bootstrap=False,n_estimators=200,min_samples_leaf=8,min_samples_split=8).fit(X_train,y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test,rf_pred)

        knn_param_grid = {'n_neighbors':[x for x in range(100)]}
        knngrid = GridSearchCV(KNeighborsClassifier(),param_grid=knn_param_grid,cv=tscv,scoring='accuracy',verbose=1,n_jobs=-1).fit(X_train,y_train)
        knn_model = KNeighborsClassifier(n_neighbors=7).fit(X_train,y_train)
        knn_pred = knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test,knn_pred)

        print('Prediction Accuracy of Google stock with the opinions approach: \n')
        print(f'SVM Model Accuracy : {100*svm_acc:.2f}%')
        print(f'RF Model Accuracy : {100*rf_acc:.2f}%')
        print(f'KNN Model Accuracy : {100*knn_acc:.2f}%')
        trade_svm = self.tradetestreturn(svm_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_rf = self.tradetestreturn(rf_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])
        trade_knn = self.tradetestreturn(knn_pred,df['Adj Close'][3072:].reset_index()['Adj Close'])

        print(f'SVM trade test Net Profit: ${trade_svm:.2f}')
        print(f'RF trade test Net Profit: ${trade_rf:.2f}')
        print(f'KNN trade test Net Profit: ${trade_knn:.2f}')
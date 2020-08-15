User Manual for running the codes: 

In any python environment import the project file and then create an object of the StockPrediction class as follows:
***
from project import StockPrediction

stock = StockPrediction() #Creating an object of the StockPrediction class inside "project.py" 


# The following comprehensive functions do the whole job, they create the raw datasets for google and apple
# and then apply the "add_all_indicators()" and "add_opinions()" functions to them to create the continuos
# technical indicators and the opinions technical indicators, and then creates SVM,RF and KNN models, trains them
# and outputs accuracy results as well as a trading simulation result for each of them using the "tradetestreturn()"
#function

stock.google_cont_models()
stock.google_op_models()
stock.apple_cont_models()
stock.apple_op_models()


# The details behind the technical indicators calculations and opinions calculations are within the "project" file
# under the StockPrediction class with a comment above each function for a quick definition
# There is also the "tradetest()" and the "tradetestreturn()" functions, the first one prints all the trades
# done during the whole testing period along with statistics of winning and losing trades and profits
# while the latter just returns the netprofit, i made it just to incorprate it with the 4 functions mentioned above

# The .h5 files within the folders hold the LSTM trained models with the saved weights, there are 4 files in total for each lstm model 
# using the cont approach and opinion approach
import pandas as pd
import pylab as pl
import numpy as np
import warnings
import pickle
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
data = pd.read_csv("carona.csv")
data = data[['fiver', 'breath', 'coughing', 'cold', 'discomfort', 'age', 'sex','infectionper']]
from sklearn.model_selection import StratifiedShuffleSplit



if __name__== "__main__":
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data,data['infectionper']):
        train= data.loc[train_index]
        test= data.loc[test_index]
    test_input = test[['fiver', 'breath', 'coughing', 'cold', 'discomfort', 'age', 'sex']].to_numpy()
    train_input = train[['fiver', 'breath', 'coughing', 'cold', 'discomfort', 'age', 'sex']].to_numpy()
    test_output = test[['infectionper']].to_numpy().reshape(len(test_input),)
    train_output = train[['infectionper']].to_numpy().reshape(len(train_input),)
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(train_input,train_output)
    
    file = open('model.pkl', 'wb')

    pickle.dump(LR, file)
    file.close()

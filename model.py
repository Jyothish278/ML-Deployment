import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
dataset=pd.DataFrame({"experience":["zero","zero","five","two","seven","three","ten","eleven"],"test_score":[8,8,6,10,9,7,8,7],"interview_score":[9,6,7,10,6,10,7,8],"salary":[50000,45000,60000,65000,70000,62000,72000,80000]})
dataset
dataset["test_score"].fillna(dataset['test_score'].mean(),inplace=True)
X=dataset.iloc[:,:3]
X
def convert_to_int(word):
    word_dict={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"zero":0,0:0}
    return word_dict[word]
X['experience']=X['experience'].apply(lambda x: convert_to_int(x))
y=dataset.iloc[:,-1]
y
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
pickle.dump(regressor,open('model.pkl','wb'))
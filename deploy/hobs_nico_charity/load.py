import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os

from sklearn.linear_model import SGDClassifier


#Variables
model = None
model_path = 'data/sgd_class_.pkl'
#edu_y_true = pd.DataFrame.from_json('data/edu.json')
edu_y_true = json.load(open('data/edu.json','rb'))
nat_y_true = json.load(open('data/nat.json','rb'))


def pipe(X):
    def func_edu_helper(level):
        try:
            return edu_y_true[level]
        except:
            return -1

    def func_nat_helper(level):
        try:
            return nat_y_true[level]
        except:
            return -1

    X_new_edu = X['education_level'].map(func_edu_helper)
    X_new_nat = X['native_country'].map(func_nat_helper)
    X['education_level'] = X_new_edu
    X['native_country'] = X_new_nat

    return X


def load_model():
    global model
    model = pickle.load(open(model_path, 'rb'))
    #model = joblib.load(model_path)


def read_json(json_file):
    df = pd.DataFrame.from_dict(json_file)
    df = df.rename(
        columns={
            'education-num': 'education_num',
            'marital-status': 'marital_status',
            'capital-gain': 'capital_gain',
            'capital-loss': 'capital_loss',
            'hours-per-week': 'hours_per_week',
            'native-country': 'native_country',
            'income': 'y'
        }
    )
    return df


def application(json_file):
    global model
    data = read_json(json_file)

    X = pipe(data)

    y_pred = model.predict(X)

    json_return = {str(index): bool(v) for index, v in np.ndenumerate(y_pred)}

    return json_return



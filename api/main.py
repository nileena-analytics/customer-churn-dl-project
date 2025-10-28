from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


app = FastAPI()

moedl = load_model('models/churn_model.h5')
scaler = pickle.load(open('models/scler.pkl', 'rb'))

class CustemerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: int
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

geo_map = {'France':0, 'Germany':1, 'Spain':2}
gender_map = {'Female':0, 'Male':1}

@app.post('/predict')
def predict_churn(features: CustemerFeatures):
    data = features.dict()
    #encoding categorical features
    data['Geography'] = geo_map.get(data['Geography'], 1)
    data['Gender'] = gender_map.get(data['Gender'], 1)

    #convert to dataframe
    X = pd.dataFrame([data])

    #Scaled
    X_scaled = scaler.transform(X)

    #prediction
    prediction = model.predict(X_scaled)
    churn_prob = prediction[0][0]
    return('churn probability:', flaot(churn_prob))

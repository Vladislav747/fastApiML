from enum import IntEnum, Enum
from typing import Optional, Any

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from joblib import load
import pandas as pd


class PClass(IntEnum):
    first = 1
    second = 2
    third = 3


class Embarked(str, Enum):
    S = 'S'
    C = 'C'
    Q = 'Q'


def to_camel(string: str) -> str:
    return ''.join(word.capitalize() for word in string.split('_'))


class Passenger(BaseModel):
    passenger_id: int
    pclass: PClass
    name: str
    sex: str
    age: Optional[float] = None
    sib_sp: int
    parch: int
    ticket: str
    fare: float
    cabin: Optional[str] = None
    embarked: Optional[Embarked] = None

    class Config:
        alias_generator = to_camel


class PassengerResponse(Passenger):
    prediction: bool


clf = load('clf_2.joblib')

app = FastAPI()


def pydantic_model_to_df(model_instance):
    return pd.DataFrame([jsonable_encoder(model_instance)])


@app.post("/predict/", response_model=PassengerResponse)
async def predict(passenger: Passenger):
    df_instance = pydantic_model_to_df(passenger)

    prediction = clf.predict(df_instance).tolist()[0]

    response = passenger.model_dump(by_alias=True)
    print(response, "response")
    response.update({'Prediction': bool(prediction)})

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
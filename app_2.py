import pickle
import pandas as pd
from typing import Optional, Any

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from joblib import load
from contextlib import asynccontextmanager


class FeatureSet(BaseModel):
    HouseAge: float
    AveRooms: float
    AveBedrooms: float
    Population: int
    AveOccupancy: float
    Latitude: float
    Longitude: float


def medinc_regressor(x: dict) -> dict:
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.DataFrame(x, index=[0])
    res = loaded_model.predict(x_df)[0]
    return {"prediction": res}


ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["medinc_regressor"] = medinc_regressor
    yield
    # вот тут очищаем все ресурсы
    ml_models.clear()


app = FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict/")
async def predict(feature_set: FeatureSet):
    return ml_models["medinc_regressor"](feature_set.model_dump())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

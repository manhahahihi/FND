# main.py

import data
import model
import utils
import uvicorn
import requests

from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from config import *

from fastapi.responses import JSONResponse


class Item(BaseModel):
    title: str
    tag: str
    content: str

class ModelPredictor:
    
    def __init__(self):

        # load model
        self.model = utils.load_model()
        # load pretrain TF-IDF
        self.truncator = utils.load_truncator()
        # load tokenizer
        self.tokenizer = utils.load_tokenizer()

    def predict(self, input: Item):

        # input preprocess
        processed_input = data.infer_input_process(input, self.truncator, self.tokenizer)

        input_ids = processed_input.input_ids.to(device)
        attention_mask = processed_input.attention_mask.to(device)

        # make predict
        with torch.no_grad():
            # Assuming input_data is your input tensor
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        prob, pred = utils.get_preds(output)

        # return predict
        return JSONResponse({
                "prediction": pred.item(),
                "prob": prob.item()
                })

class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post(f"/predict")
        async def predict(input: Item, request: Request):
            
            response = self.predictor.predict(input)

            return response

    def get_app(self):
        return self.app

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":


    predictor_1 = ModelPredictor()
    
    api = PredictorApi(predictor_1)

    api.run(port=api_port)

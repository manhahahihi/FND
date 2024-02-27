# main.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import requests
from pydantic import BaseModel
import uvicorn
import json
from config import *
import csv
import pandas as pd
from fuzzywuzzy import fuzz  # pip install fuzzywuzzy



class Item(BaseModel):
    title: str
    tag: str
    content: str

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict")
async def predict(request: Request, title: str = Form(...), tag: str = Form(...), content: str = Form(...)):

    input = {"title": title,
             "tag": tag,
             "content": content}
    
    # print(title, tag, content, path_data)

    # check if input is existed in the dataset
    content_existed, label = check_if_exists_with_fuzzy_matching(title, tag, content, path_data)
    # print(content_existed, label)

    if content_existed:
        if label == 0:
            prediction = "Based on a match in our dataset, our system predicts this is REAL news story."
        elif label == 1:
            prediction = "Based on a match in our dataset, our system predicts this is FAKE news story."
    else:   
        prediction = get_prediction(input)

    return templates.TemplateResponse("index.html", {"request": request, \
                                                      "news_title": title, \
                                                      "news_tag": tag, \
                                                      "news_content": content, \
                                                    "prediction": prediction})

def run(app, port):
    uvicorn.run(app, host="0.0.0.0", port=port)

def get_prediction(input: Item):
    url = api_url

    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(input)
    response = requests.post(url, data=data_json, headers=headers)
    res = response.json()
    
    if response.status_code == 200:
        if res['prediction'] == 0:
            res = f"Our advanced analysis deems this information trustworthy despite not being in our dataset."
        else:
            res = f"Our advanced analysis indicates this information isn't in our current dataset, potentially flagged as incorrect or deceptive."
        return res
    else:
        return None

def check_if_exists_with_fuzzy_matching(title, tag, content, file_path, threshold=80):
    """Checks if the given input values exist in a large CSV or XLSX file,
    using fuzzy matching to handle potential variations in the data.

    Args:
        title (str): The title to check.
        tag (str): The tag to check.
        content (str): The content to check.
        file_path (str): The path to the CSV or XLSX file.
        threshold (int): The minimum similarity score for fuzzy matching (default: 80).

    Returns:
        bool: True if matching values are found in the file, False otherwise.
        label (int): groundtruth label in the database
    """

    try:
        if file_path.endswith(".csv"):
            for chunk in pd.read_csv(file_path, chunksize=100000):
                for index, row in chunk.iterrows():
                    if (fuzz.ratio(row['Title'], title) >= threshold and
                        # fuzz.ratio(row['Tag'], tag) >= threshold and
                        fuzz.ratio(row['Content'], content) >= threshold):
                        return True, row['Label']
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
            for index, row in df.iterrows():
                if (fuzz.ratio(row['Title'], title) >= threshold and
                    # fuzz.ratio(row['Tag'], tag) >= threshold and
                    fuzz.ratio(row['Content'], content) >= threshold):
                    return True, row['Label']
                
        return False, None

    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error: {e}")
        return False, None 

    
if __name__ == "__main__":
    run(app, web_port)

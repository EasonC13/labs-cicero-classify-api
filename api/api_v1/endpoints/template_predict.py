from fastapi import APIRouter, Depends, status, Response

from typing import Optional

from pydantic import BaseModel

from typing import Any, Dict, AnyStr, List, Union

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

TAG_OF_TEMPLATE_CLASSIFY_API = ["Predict"]

from utils.template_classification import model
class classify_template_body(BaseModel):
    return_max_size = 20
    text = model.example_text


@router.post("/models/classify/template", tags = TAG_OF_TEMPLATE_CLASSIFY_API)
def classify_template(response: Response, data: classify_template_body):
    predictions = model.classify(data.text)
    if data.return_max_size > 0:
        predictions = predictions[:data.return_max_size]
    response.status_code = status.HTTP_200_OK
    return {
        "message": "get success",
        "prediction": predictions
    }



@router.get("/models/classify/template", tags = TAG_OF_TEMPLATE_CLASSIFY_API)
def get_classify_model_status(response: Response):
    response.status_code = status.HTTP_200_OK
    return {
        "message": "get success",
        "status": "online"
        }

from pymongo import MongoClient
from core.config import (
    MONGODB_URL,
    DATABASE_NAME,
    LABEL_COLLECTION,
    API_PORT,
    API_HOST,
)
import json
import re
import requests
import os

def test_adapter_available(target_label):
    status = requests.get(f"http://{API_HOST}:{API_PORT}/api/v1/models/NER/labelText")
    status = status.json()
    assert target_label["label_name"] in status["label"]
    assert target_label["adapter"]["current_filename"] in status["version"]
    return True

def test_prediction_work_with_specific_label_then_return_predict_confidence(target_label):
    data = {
      "return_top_n": -1,
      "return_higher_than": 0,
      "wanted_labels": [
        target_label["label_name"]
      ],
      "text": "Dan will be deemed to"
    }
    result = requests.post(f"http://{API_HOST}:{API_PORT}/api/v1/models/NER/labelText", data = json.dumps(data))

    assert result.status_code == 200
    assert result.json()["prediction"][0]["predictions"][0]["type"] == target_label["label_name"]
    return result.json()["prediction"][0]["predictions"][0]["confidence"]

def test_adapter_unavailable(target_label):
    status = requests.get(f"http://{API_HOST}:{API_PORT}/api/v1/models/NER/labelText")
    status = status.json()
    assert target_label["label_name"] not in status["label"]
    assert target_label["adapter"]["current_filename"] not in status["version"]
    return True

def test_prediction_cant_work_with_unknown_label(target_label):
    data = {
      "return_top_n": -1,
      "return_higher_than": 0,
      "wanted_labels": [
        target_label["label_name"]
      ],
      "text": "Dan will be deemed to"
    }
    result = requests.post(f"http://{API_HOST}:{API_PORT}/api/v1/models/NER/labelText", data = json.dumps(data))

    assert result.status_code == 404, ""
    return True
client = MongoClient(MONGODB_URL)

label_col = client[DATABASE_NAME][LABEL_COLLECTION]

# Find an label which it's adapter filename is not empty.
target_label = label_col.find_one({
    "adapter.current_filename": re.compile("^(?!\s*$).+"),
    "$where": "this.adapter.history.length > 1",
})

origin_target_label = target_label.copy()

try:
    test_adapter_available(target_label)
    origin_confidence = test_prediction_work_with_specific_label_then_return_predict_confidence(target_label)
    label_col.update_one({
        "_id": target_label["_id"]
    }, {
        "$set": {
            "adapter.current_filename": ""
        }
    })
    test_adapter_unavailable(target_label)
    test_prediction_cant_work_with_unknown_label(target_label)
    for history in target_label["adapter"]["history"]:
        if history["filename"] != target_label["adapter"]["current_filename"]:
            break
    

    label_col.update_one({
        "_id": target_label["_id"]
    }, {
        "$set": {
            "adapter.current_filename": history["filename"]
        }
    })
    
    # TBA: A API that can specify the version when predict
    
    test_adapter_available(target_label)
    new_confidence = test_prediction_work_with_specific_label_then_return_predict_confidence(target_label)
    assert new_confidence != origin_confidence
except Exception as e:
    print(e)
finally:
    label_col.update_one({
        "_id": target_label["_id"]
    }, {
        "$set": {
            "adapter.current_filename": origin_target_label["adapter"]["current_filename"]
        }
    })

try:
    print(f"Finish test {os.path.basename(__file__)}")
except:
    pass
from inspect import trace
from fastapi import APIRouter, Depends, status, Response

from typing import Optional

from pydantic import BaseModel

from core.config import (
    ALLOWED_HOSTS,
    PROJECT_NAME,
    PROJECT_VERSION,
    API_PORT,
    DATABASE_NAME,
    NER_LABEL_COLLECTION,
    Feedback_Template_Collection,
    Feedback_Suggestion_Collection,
    LABEL_COLLECTION,
    LABEL_TRAIN_JOB_COLLECTION,
)
from db.mongodb import AsyncIOMotorClient, get_database
from bson.objectid import ObjectId
import asyncio
from typing import Any, Dict, AnyStr, List, Union
from datetime import datetime
from db.utils import convert_mongo_id
from utils.trainer_communicate import asyncio_update_db_last_modify_time, set_trainer_restart_required
import re
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()
LABEL_API_TAGS = ["Label"]

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."

class create_new_label_body(BaseModel):
    user: str = "example@gmail.com"
    label_name: str = "Party"
    inherit: list = ["B-per", "I-per", "B-org", "I-org"]
    alias_as: list = ["String"]
    comment: str = """This is the example cased."""
    tags: list = []


@router.post("/labels", tags = LABEL_API_TAGS, status_code=status.HTTP_200_OK)
async def define_new_label(response: Response, data: create_new_label_body):
    result = re.findall(r'[;|(|)]',data.label_name)
    if len(result) != 0:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return {
            "message": "Label Name must not contain \"" + '\", \"'.join(result) + '"'
        }

    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]
    result = col.find({"label_name": data.label_name})
    result = await result.to_list(None)
    if len(result) != 0:
        result[0]["id"] = str(result[0]["_id"])
        del result[0]["_id"]
        return {
            "message": f"Failed, Already have {data.label_name}",
            "label": result[0]
        }
    else:
        dataToStore = {
            "user": data.user,
            "label_name": data.label_name,
            "inherit": data.inherit,
            "alias_as": data.alias_as,
            "comment": data.comment,
            "tags": data.tags,
            "adapter": {
                "current_filename": "",
                "training_status": "",
                "history": [],
                "update_time": datetime.now(),
            },
            "create_time": datetime.now(),
        }
        try:
            result = await col.insert_one(dataToStore)
            response.status_code = status.HTTP_201_CREATED
            return {
                "message": f"Success, new label {data.label_name} added."
            }
        except Exception as e:
            return {
                "message": f"Fail, please check the Error Message",
                "error_msg": str(e)
            }

@router.get("/labels", tags = LABEL_API_TAGS)
async def get_all_label():
    mongo_client = await get_database()
    label_define_col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]

    labels = await label_define_col.find({}, {"_id": False}).to_list(None)
    return labels



@router.get("/labels/{label_name}", tags = LABEL_API_TAGS)
async def get_label_by_name(label_name, response: Response):
    mongo_client = await get_database()
    label_define_col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]
    label_data_col = mongo_client[DATABASE_NAME][NER_LABEL_COLLECTION]

    label = await label_define_col.find_one({"label_name": label_name})
    if label == None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "message": "Failed, Can't find this label name"
        }
    label = convert_mongo_id(label)

    texts = label_data_col.find(
        {"text_and_labels.labels": {"$in": [label_name] + label["inherit"]}},
        {"text_and_labels": False}
    )
    texts = await texts.to_list(None)
    texts_count = len(texts)

    label["data_count"] = texts_count

    label["description(auto_generated)"] = f"""Label "{label["label_name"]}" is to label out {label["label_name"]} in concerto contract. {label["label_name"]} also all the {label["inherit"]} labels in current and future dataset, and when training {label["alias_as"]}, text which labeled as Party will also be labeled positively. Currently, we have {label["data_count"]} datas contain this label in training dataset."""
    return label

class update_data_body(BaseModel):
    user: str = "example@gmail.com"
    tags: Optional[list] = [""]
    texts: JSONArray = [
        {
            "text": "Eason",
            "labels": ["Party", "String"]
        },
        {
            "text": "will",
            "labels": ["O"]
        },
        {
            "text": "meet",
            "labels": ["O"]
        },
        {
            "text": "Dan",
            "labels": ["Party", "String"]
        },
        {
            "text": "at",
            "labels": ["O"]
        },
        {
            "text": "2021-08-04 18:00",
            "labels": ["TemporalUnit"]
        },
        {
            "text": ".",
            "labels": ["O"]
        },
    ]

from utils.tokenizer import tokenizer as roberta_tokenizer

@router.post("/data/labeledText", tags = LABEL_API_TAGS, status_code=status.HTTP_200_OK)
async def update_labeled_data(response: Response,
                              data: update_data_body,
                              refreash_trainer: bool = False):
    sentences = []
    current_sentence = []
    # Split the texts by dot. 
    # So avoid CUDA Out of Memory at training by too long texts.
    # This won't affect NER model's performance.
    for text in data.texts:
        current_sentence.append(text)
        if text == {'text': '.', 'labels': ['O']}:
            # New Sentence, New data.
            sentences.append(current_sentence)
            current_sentence = []
    sentences.append(current_sentence)
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][NER_LABEL_COLLECTION]
    insert_ids = []
    for sentence in sentences:
        token_and_labels = []
        last_word_index = len(sentence)-1
        for i, text in enumerate(sentence):
            if i != 0 and i != last_word_index:
                text_to_do = " " + text["text"]
            else:
                text_to_do = text["text"]
            tokens = roberta_tokenizer.tokenize(text_to_do)
            for j, token in enumerate(tokens):
                token_and_labels.append({
                    "token": token,
                    "labels": text["labels"]
                })
        dataToStore = {
            "user": data.user,
            "tags": data.tags,
            "text_and_labels": sentence,
            "token_and_labels": token_and_labels,
            "TimeStamp": datetime.now(),
        }
        # todo: Check the label in text all included.
        result = await col.insert_one(dataToStore)
        insert_ids.append(str(result.inserted_id))
    await asyncio_update_db_last_modify_time(NER_LABEL_COLLECTION)

    if refreash_trainer:
        await set_trainer_restart_required(True)
    return {
        "message": "Add Success",
        "insert_ids": insert_ids
    }


@router.get("/data/labeledText", tags = LABEL_API_TAGS)
async def get_labeled_data(response: Response,
                           label_name: str = None,
                           detail: bool = False, 
                           start: int = 0, 
                           end: int = 10):
    if end == -1 and start == -1: end = None
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][NER_LABEL_COLLECTION]
    
    result = col.find({"text_and_labels.labels": {"$in": [label_name]}},
                      {"text_and_labels": detail, 
                       "token_and_labels": detail})
    
    result = await result.to_list(end)
    result = result[start:end]
    result = list(map(convert_mongo_id,result))
    response.status_code = status.HTTP_200_OK
    return {
        "message": "Success",
        "data": result
    }

class custom_filter(BaseModel):
    mongo_filter: dict = {"user": "example@gmail.com"}

@router.post("/data/labeledText/find:by:mongo:filter", tags = LABEL_API_TAGS)
async def get_labeled_data_by_custom_filter(response: Response,
                           data: custom_filter,
                           detail: bool = False, 
                           start: int = 0, 
                           end: int = 10):
    if end == -1 and start == -1: end = None
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][NER_LABEL_COLLECTION]
    
    result = col.find(data.mongo_filter,
                      {"text_and_labels": detail, 
                       "token_and_labels": detail})
    
    result = await result.to_list(end)
    result = result[start:end]
    result = list(map(convert_mongo_id,result))
    response.status_code = status.HTTP_200_OK
    return {
        "message": "Success",
        "data": result
    }

@router.get("/data/labeledText/{_id}", tags = LABEL_API_TAGS)
async def get_labeled_data_by_id(response: Response,
                                 _id, 
                                 detail: bool = True):
    
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][NER_LABEL_COLLECTION]
    
    result = await col.find_one({"_id": ObjectId(_id)},
                                {"text_and_labels": detail, 
                                 "token_and_labels": detail})
    if result:
        result["_id"] = str(result["_id"])
        response.status_code = status.HTTP_200_OK
        return {
            "message": "Success",
            "data": result
        }
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "message": f"_id {_id} Not Found."
        }

@router.delete("/data/labeledText/{_id}", tags = LABEL_API_TAGS)
async def delete_labeled_data_by_id(response: Response, _id):
    
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][NER_LABEL_COLLECTION]
    
    result = await col.delete_one({"_id": ObjectId(_id)})
    if result.deleted_count:
        response.status_code = status.HTTP_204_NO_CONTENT
        return response
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        response.status_code = status.HTTP_200_OK
        return {
            "message": f"Failed, {_id} Not Found."
        }
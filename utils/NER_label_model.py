import pandas as pd
import numpy as np

import torch
import transformers
# When Development
# Torch Version: 1.8.2+cu111
# Transformers (Adapter) Version: 2.0.1
print(f"Torch Version: {torch.__version__}")
print(f"Transformers (Adapter) Version: {transformers.__version__}")

print(f"Loading adapter model...")

import numpy as np



from transformers import RobertaTokenizer
from utils.tokenizer import tokenizer

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

from utils.ner_dataset import get_trainset_data_loader

from transformers import RobertaConfig, RobertaModelWithHeads

from utils.logger.utils import mute_logging
with mute_logging():
    config = RobertaConfig.from_pretrained("roberta-base")
    model = RobertaModelWithHeads.from_pretrained(
        "roberta-base",
        config=config,
    )

import os
import re

# This is a asyncio status, but just use mongo client directly to save develop time
from pymongo import MongoClient
from core.config import (
    MONGODB_URL,
    DATABASE_NAME,
    LABEL_COLLECTION,
    NER_ADAPTERS_PATH,
    PREDICT_DEVICE,
)


def check_adapter_filename_valid(filename):
    if (os.path.isdir(f"{NER_ADAPTERS_PATH}/save_adapters/{filename}") and
        os.path.isdir(f"{NER_ADAPTERS_PATH}/save_heads/{filename}")):
        return True
    else:
        return False

mongo_client = MongoClient(MONGODB_URL)
labels_col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]

def get_label_adapter_filenames():
    all_adapters = {}
    labels = labels_col.find()
    labels = list(labels)
    for label in labels:
        if label["adapter"]["current_filename"]:
            filename = label["adapter"]["current_filename"]
            if check_adapter_filename_valid(filename) == False:
                while len(label["adapter"]["history"]) > 0:
                    hisotry_adapter = label["adapter"]["history"].pop(-1)
                    filename = hisotry_adapter["filename"]
                    if check_adapter_filename_valid(filename):
                        print(f"""Label {label["label_name"]} will use a history one "{filename}" because current one unavailable.""")
                        break
            all_adapters[label["label_name"]] = filename
    return all_adapters

global_adapters_filenames = get_label_adapter_filenames()


def load_adapters(adapters_filenames):
    global model
    all_adapter_name = []
    with mute_logging():
        for adapter_filename in adapters_filenames:
            name = model.load_adapter(f"{NER_ADAPTERS_PATH}/save_adapters/{adapter_filename}")
            all_adapter_name.append(name)
            model.load_head(f"{NER_ADAPTERS_PATH}/save_heads/{adapter_filename}")
    return all_adapter_name
global_all_adapter_names = load_adapters(global_adapters_filenames.values())


from transformers.adapters.composition import Parallel

def update_model_active_head(adapter_names = global_all_adapter_names):
    global model
    parallel = Parallel(*adapter_names)
    model.set_active_adapters(parallel)

update_model_active_head(global_all_adapter_names)


def check_and_update_adapter():
    """If there is difference, then update."""
    global global_adapters_filenames
    global global_all_adapter_names
    update = get_label_adapter_filenames()
    new_adapters = []
    for label_name, filename in update.items():
        if (label_name not in global_adapters_filenames.keys() or
            filename not in global_adapters_filenames.values()):
            global_adapters_filenames[label_name] = filename
            new_adapters.append(filename)
    
    if len(new_adapters) != 0:
        new_adapter_names = load_adapters(new_adapters)
        global_all_adapter_names = set(update.keys())
        update_model_active_head(global_all_adapter_names)
    return True

def have_adapter_version(label_name, adapter_filename):
    print(label_name, adapter_filename)
    label = labels_col.find_one(
        {"label_name": label_name},
        {"adapter": True})
    result = filter(lambda x: x["filename"] == adapter_filename,
                    label["adapter"]["history"])
    result = list(result)
    return bool(result)

def check_adapters_version_available(version_map = {}):
    check_pass = True
    failed = {}
    for label_name, adapter_version in version_map.items():
        if have_adapter_version(label_name, adapter_version) == False:
            failed[label_name] = adapter_version
            check_pass = False

    return (check_pass, failed)

def update_global_adapters_filenames(version_dict):
    for key, value in version_dict.items():
        global_adapters_filenames[key] = value

def predict(sentence, device = PREDICT_DEVICE):
    global model
    tokenized_sentence = torch.tensor([tokenizer.encode(sentence)])
    pos = torch.tensor([[0] * len(tokenized_sentence)])
    tags = torch.tensor([[1] * len(tokenized_sentence)])

    model = model.to(device)
    with torch.no_grad():
        outputs = model(input_ids=tokenized_sentence.to(device), 
                        token_type_ids=pos.to(device), 
                        attention_mask=tags.to(device))

    logits = outputs[1][0]

    return_tags_order = {}
    all_output = None
    for i, output in enumerate(outputs):

        return_tags_order[i] = (model.active_head[i])

        output = outputs[i][0]

        if all_output != None:
            all_output = torch.cat((all_output, output), dim=2)
        else:
            all_output = output
    all_output = torch.sigmoid(all_output)

    output_array = np.array(all_output)
    output_array = output_array.reshape(output_array.shape[-2], output_array.shape[-1])

    label_confidences = []
    for label_confidence in list(output_array):
        label_confidences.append(list(label_confidence.astype(float)))

    #Drop Head and End since it is start/stop Token
    label_confidences = label_confidences[1:-1]

    max_value = np.array(label_confidences).argmax(axis=1)
    trans_func = np.vectorize(lambda x: model.active_head[x])
    out_labels = trans_func(max_value)
    out_labels = list(out_labels)

    out_sentence = tokenizer.tokenize(sentence)

    return out_sentence, out_labels, label_confidences, return_tags_order
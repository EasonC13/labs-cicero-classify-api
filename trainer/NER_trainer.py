import pandas as pd
import pymongo
import numpy as np
import torch
from utils.logger.utils import mute_logging
import time
from core.config import (
    NER_ADAPTERS_TRAINER_NAME,
    MONGODB_URL,
    DATABASE_NAME,
    NER_LABEL_COLLECTION,
    LABEL_COLLECTION,
    LABEL_TRAIN_JOB_COLLECTION,
    LABEL_TRAIN_JOB_COLLECTION,
    NER_TRAIN_BATCH_SIZE,
    NER_TRAIN_DEFAULT_FILTER,
    NER_TRAIN_DEVIDE_ID,
    PATH,
    NER_ADAPTERS_PATH,
)
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

from utils.trainer.NER import get_training_dataframe, NER_Dataset_for_Adapter
import re
import sys
import datetime
from utils.trainer_communicate import update_pid

import os
# When Each Train
# Run When Set Up
update_pid(NER_ADAPTERS_TRAINER_NAME, os.getpid())
if os.path.isdir(f"{PATH}/{NER_ADAPTERS_PATH}/save_adapters") == False:
    os.mkdir(f"{PATH}/{NER_ADAPTERS_PATH}/save_adapters")
if os.path.isdir(f"{PATH}/{NER_ADAPTERS_PATH}/save_heads") == False:
    os.mkdir(f"{PATH}/{NER_ADAPTERS_PATH}/save_heads")



device = torch.device(f"cuda:{NER_TRAIN_DEVIDE_ID}" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.logs import trainer_log, queue_task_log

client = pymongo.MongoClient(MONGODB_URL)
training_job_col = client[DATABASE_NAME][LABEL_TRAIN_JOB_COLLECTION]

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 
    if samples[0][2] is not None:
        label_ids = [s[2] for s in samples]
        label_ids = pad_sequence(label_ids, 
                                    batch_first=True)
    else:
        label_ids = None

    # zero pad to same length
    tokens_tensors = pad_sequence(tokens_tensors, 
                                    batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)

    # attention masks, set zero padding in tokens_tensors
    # to 1 so BERT only attention on those tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def train_model_with_auto_adjust_batch(model, i, data, now_is_training, label_index):
    """Use divide and conquer to avoid CUDA out of memory error (OOM).
    If Out of memory, try again with half of the batch size.
    If OOM again, try again with half and half of the batch, etc."""
    datas = [data]
    while len(datas) != 0:
        data = datas[0]
        try:
            # Train
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]
            outputs = model(input_ids = tokens_tensors,
                attention_mask=masks_tensors,
                token_type_ids=segments_tensors)
            
            logits = outputs[0]

            current_label = labels.view(-1, labels.shape[-1])[:, label_index]
            current_label = current_label.view(-1)

            active_logits = logits.view(-1, logits.shape[-1])[masks_tensors.view(-1) == 1]

            actual = current_label[masks_tensors.view(-1)== 1].float().view(-1,1)

            loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(active_logits, actual)
            loss.backward()
            datas.pop(0)
        except Exception as error:
            if "CUDA" not in error.args[0]:
                raise error
            # del the variables are unnecessary, Python can check automatically.
            # So I comment it in case anyone want to do the same thing here :D
            #
            # del tokens_tensors, segments_tensors, \
            # masks_tensors, labels
            # torch.cuda.empty_cache()
            msg = f"Failed, CUDA out of memory, dividing data from shape {np.array(datas[0][0]).shape}"
            queue_task_log(now_is_training["_id"], msg)
            length = len(datas[0][0])
            half = int(length/2)
            if length != 1:
                devided_datas = []
                for d in datas:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = d
                    
                    devided_datas.append((tokens_tensors[:half], segments_tensors[:half], \
                    masks_tensors[:half], labels[:half]))
                    
                    devided_datas.append((tokens_tensors[:half], segments_tensors[:half], \
                    masks_tensors[:half], labels[:half]))
                datas = devided_datas
        finally:
            # do empty_cache every run to avoid CUDA OOM,
            # and let other programs can use GPU, too.
            torch.cuda.empty_cache()
    # When testing with Batch Size 128.
    # without clean cache, cost: 01:33 per 100 iteration.
    # with clean cache directly, cost: 01:43 per 100 iteration.
    # with threading clean cache and join, cost: 01:44 per 100 iteration.
    # with threading clean cache without join, cost: 01:44 per 100 iteration.
    ## Therefore, I clean cache directly in this case.
    return loss
    

def get_target_df_by_filter(df, train_data_search_filter):
    client = pymongo.MongoClient(MONGODB_URL)
    col = client[DATABASE_NAME][NER_LABEL_COLLECTION]

    wanted_id = list(map(lambda x: str(x["_id"]),
                            list(col.find(train_data_search_filter, {"_id": True}))))

    target_df = df[df["Sentence #"].isin(wanted_id)]
    return target_df

def main():
    try:
        df = get_training_dataframe(NER_TRAIN_DEFAULT_FILTER)    
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        while True:
            dateStamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
            training_queue = training_job_col.find({
                "status":  re.compile("(training|waiting)")
            }, {"logs": False})
            training_queue = list(training_queue)

            if len(training_queue) == 0:
                import sys
                sys.exit(0)

            # Old Train First
            training_queue.sort(key = lambda x: x["add_time"], reverse= False)

            now_is_training = training_queue[0]

            # Update parameter On Each Iter
            train_data_search_filter = now_is_training["train_data_filter"]
            label_name = now_is_training["label_name"]
            Epoch_Times = now_is_training["epochs"]

            print(f"""Prepare to train {label_name} with trace id {str(now_is_training["_id"])}""")

            target_df = get_target_df_by_filter(df, train_data_search_filter)
            trainset = NER_Dataset_for_Adapter(tokenizer, target_df, label_name)

            log_msg = f"Start training {label_name} with batch_size={NER_TRAIN_BATCH_SIZE} and epoch={Epoch_Times}, have {len(training_queue) -1} in the waiting line..."
            trainer_log(log_msg)
            queue_task_log(now_is_training["_id"], log_msg)

            label_define_col = client[DATABASE_NAME][LABEL_COLLECTION]
            label_define_col.update_one(
                {"label_name": label_name},
                {"$set": {
                    "adapter.training_status": "training",
                    }
                })
            now_is_training_label_defined = label_define_col.find_one({"label_name": label_name})
            training_job_col.update_one({
                    "_id": now_is_training["_id"],
                },{
                    "$set": {
                        "status": "training",
                        "train_data_count": len(trainset),
                        "positive_label": trainset.positive_label,
                        "label_snapsnot": {
                            "_id": str(now_is_training_label_defined["_id"]),
                            "user": now_is_training_label_defined["user"],
                            "label_name": now_is_training_label_defined["label_name"],
                            "inherit": now_is_training_label_defined["inherit"],
                            "alias_as": now_is_training_label_defined["alias_as"],
                            "comment": now_is_training_label_defined["comment"],
                            "tags": now_is_training_label_defined["tags"],
                        }
                    }
                })

            

            trainloader = DataLoader(trainset, batch_size=NER_TRAIN_BATCH_SIZE, 
                                    collate_fn=create_mini_batch)

            from transformers import RobertaConfig, RobertaModelWithHeads
            config = RobertaConfig.from_pretrained(
                "roberta-base"
            )

            try:
                with mute_logging():
                    model = RobertaModelWithHeads.from_pretrained(
                        "roberta-base",
                        config=config,
                        )

                
                model.add_adapter(label_name)
                model.add_tagging_head(
                    label_name,
                    num_labels=1
                    )
                
                model.train_adapter(label_name)
                model = model.to(device)

                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                                {
                                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                    "weight_decay": 1e-5,
                                },
                                {
                                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                    "weight_decay": 0.0,
                                },
                            ]
                optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=1e-4)

                for epoch in range(Epoch_Times):
                    epoch += 1 # epoch start from 1
                    print(f"\n{label_name} epoch {epoch} start")
                    start_time = datetime.datetime.now()
                    label_index = trainset.label_map[label_name]
                    for i, data in enumerate(trainloader):
                        loss = train_model_with_auto_adjust_batch(model, i, data, now_is_training, label_index)
                        if i % 10 == 0:
                            # with threading to push log onto db, cost: 0:01:43.776396 per 100 iteration.
                            # without threading to push log onto db, cost: 0:01:43.495900 per 100 iteration.
                            msg = f"[At Epoch {epoch} Round {i}] Loss: {loss}"
                            queue_task_log(now_is_training["_id"], msg)
                        optimizer.step()
                        optimizer.zero_grad()
                    print(f"{label_name} epoch {epoch} end, this epoch cost {datetime.datetime.now() - start_time}")
                print("Finish, Saving")
                filename = f"{label_name}_epoch_{Epoch_Times}_{dateStamp}"
                model.save_adapter(f"{NER_ADAPTERS_PATH}/save_adapters/{filename}", model.active_adapters[0])
                model.save_head(f"{NER_ADAPTERS_PATH}/save_heads/{filename}", model.active_head)
            except Exception as error:
                import traceback
                import sys
                exc_type, exc_value, exc_tb = sys.exc_info()
                result = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                print(result)
                trainer_log(result)
                queue_task_log(now_is_training["_id"], result)
                raise error

            training_job_col.update_one({
                    "_id": now_is_training["_id"],
                },{
                    "$set": {
                        "store_filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                        "status": "done",
                    }
                })

            label_define_col = client[DATABASE_NAME][LABEL_COLLECTION]
            now_time = datetime.datetime.now()
            label_define_col.update_one(
                {"label_name": label_name},
                {"$set": {"adapter.current_filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                        "adapter.training_status": "done",
                        "adapter.update_time": now_time,
                    },
                "$push": {"adapter.history": {
                        "filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                        "time": now_time,
                        "trainer_job_id": str(now_is_training["_id"]),
                    }}})
    except KeyboardInterrupt:
        import sys
        sys.exit(1)
    except Exception as error:
        import traceback
        import sys
        exc_type, exc_value, exc_tb = sys.exc_info()
        result = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(result)
        trainer_log(result)
        queue_task_log(now_is_training["_id"], result)
        raise error

if __name__ == "__main__":
    main()
    sys.exit(0)
# labs-cicero-classify-api

- [How we design the NER model with Adapter?](./docs/How_we_design_the_NER_model.md).
- [How Backend Train and Use NER Adapters?](./docs/How_backend_train_and_use_NER_Adapters.md)

# How to host API?

> We strongly suggest hosting the API at the Ubuntu with at least 4 cores, 8GB Ram, and a GPU with more than 8 GB VRAM. Yet you also can host it without GPU or on other systems. Welcome to share your successful installation step on other system here.

### 0. Install dependency

A. [Install Anaconda following this link](https://docs.anaconda.com/anaconda/install/index.html).

B. API requires MongoDB as the database. Please [install MongoDB following this guide](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/), or use an existing host you have.

C. Install CUDA version 11.1 (for PyTorch 1.8.2 LTS)
<!-- Tutorial of CUDA is TBA -->
If you want to check your CUDA version, please try the following commands:

```
/usr/local/cuda/bin/nvcc --version
```

If you see the output like this, you have CUDA 11.1.

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
```

### 1. Create Python Environment

####  Clone the repository:

```
git clone https://github.com/EasonC13/labs-cicero-classify-api.git
cd labs-cicero-classify-api
```

#### Create a new clean environment

Create an environment. You can change the following name from "adapter" to anything you want.

```
conda create --name adapter python=3.7
```

> [Adapter Transformers](https://pypi.org/project/adapter-transformers/) is a fork of [Transformers](https://pypi.org/project/transformers/), and will overlap the Transformer library when you import. Therefore, I strongly suggest you create an isolated environment for Adapter Transformer aside from Transformers to avoid potential conflict or error.

```
conda activate adapter
```

Install PyTorch. Below is the script for install PyTorch version 1.8.2 at CUDA 11.1, if you use other CUDA version, please [check this guide](https://pytorch.org/get-started/locally/#start-locally)).

```
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

Install the other requirements

```
pip install -r requirements.txt
```

### 2. Configuration

At the **root path** of the repo (where `start.py` located), run the following command:

```
vim .env
```

Then put your MongoDB user and password inside.

```
MONGODB_PORT=27017
MONGODB_HOST="localhost"
MONGO_USER="your_username"
MONGO_PASSWORD="your_password"
```

Then go to `core/config.py`

```
vim core/config.py
```

There are some things I think you might need to change in config:

```
DATABASE_NAME = "Accord_Project"

NER_TRAIN_BATCH_SIZE = 256

ANACONDA_ENV_NAME = "adapter"

```

### 3. Run API and trainer

Just run the following command at your command line:

```
python start.py
```

Then the `start.py` will run uvicorn to host the FastAPI backend, and run `trainer/NER_trainer_runner.py` to start the NER model trainer.

After service start is finish, you can go to [http://localhost:13537/docs](http://localhost:13537/docs) to see all available API and its documents. (Replace localhost with your IP)

![](https://i.imgur.com/OjobB6P.png)

### Setup operation after run
Please download the example NER data (TBA).

And run the following command on repo's root path.

```
jupyter notebook Setup_Operation
```

then run the following three ipynb script to add the example training data into DB and start Training your first NER labels by API.

- [post_defined_labels_onto_DB_by_API.ipynb](./Setup_Operation/post_defined_labels_onto_DB_by_API.ipynb)
- [post_train_data_to_db.ipynb](./Setup_Operation/post_train_data_to_db.ipynb)
- [put_all_labels_into_training_job.ipynb](./Setup_Operation/put_all_labels_into_training_job.ipynb)

### Develop and Debug

If you are using VScode, please go to `.vscode/setting.json`, then change the paths to your python path. (If you use Anaconda, just change `adapter` to your conda env name.)

```
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/adapter/bin/python",
    "python.analysis.extraPaths": [
        "~/anaconda3/envs/adapter/lib/python3.7/site-packages/"
    ],
    "python.autoComplete.extraPaths": [
        "~/anaconda3/envs/adapter/lib/python3.7/site-packages/"
    ]
}
```

Then press F5 into debug mode. This will trigger `.vscode/launch.json` to start API backend as debug mode. Then go to the port it show to debug (default is 13537).

In debug mode, every file you save will reload the API backend service. 

If you feeling the loading time is too long (because ML models need some time). you can go to `api/api_v1/api.py` and comment the following three router.

```
from .endpoints.NER_label_predict import router as NER_label_predict_router
router.include_router(NER_label_predict_router)

from .endpoints.template_predict import router as template_predict_router
router.include_router(template_predict_router)

from .endpoints.suggestion_predict import router as suggestion_predict_router
router.include_router(suggestion_predict_router)
```

Utils the API docs http://localhost:13537/docs to test and debug the API.


# Unit Test
The unit test of the API is not finished yet. (Welcome PR)

Now you only can use the following example one to test the NER label change version feature is workable.

```
python test/Dynamic_import_and_remove_adapter_in_real_time.py
```
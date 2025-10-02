Step1: Create python-env or conda-env and doing setup
---
- python -m venv env
- source source env/bin/activate
- pip install seaborn pandas transformers seqeval tensorboard tensorflow
- cd ..
- pip install -e ./NVFlare/
- pip install -e ./PETINA/
- Depend on your system, you can install the Pytorch module of Rocm that you want. I installed the pytorch rocm 5.6
- pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
- You can access `/python_env/requirements.txt` to look at my env to compare.
- Inside this NVFlare already change the gpu_utils to what we are using. You can take a look at `NVFlare/nvflare/fuel/utils/gpu_utils.py`

Step2: download model and dataset
---
Since I have limitation in my home folder, so I choose the  hugging face folder in another place
- export HF_HOME="./hf_home"
- export HF_HUB_DISABLE_TELEMETRY=1
- mkdir -p "$HF_HOME"
- If you download the model, it will look like this
![Download model](./resource/1.png)
- Dataset can be found at `experiment_folder/data/nlp-ner`
- Each split will have dataset for each client. Each client will have a train and test dataset.

Step3: Edit the datapath
---
- Edit job-folder path. 
- Get the path of your data folder. In my case it is `/lustre/orion/csc666/proj-shared/ducnguyen/new_nvflare_edge/AMD_NVFlare/experiment_folder/data`
- go into each `config_fed_client.json` and change `DATASET_ROOT` value. 

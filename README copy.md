Here’s a ready-to-use Markdown file content. You can save it as `NVFlare_PETINA_Setup.md`:

````markdown
# NVFlare + PETINA Setup & Run Guide

## Step 1: Create Python Environment

1. Navigate to the environment folder:
   ```bash
   cd python_env/
    ````

2. Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate
   ```
3. Install necessary Python packages:

   ```bash
   pip install seaborn pandas transformers seqeval tensorboard tensorflow
   ```
4. Go back to the main folder:

   ```bash
   cd ..
   ```
5. Install NVFlare and PETINA in editable mode:

   ```bash
   pip install -e ./NVFlare/
   pip install -e ./PETINA/
   ```
6. Install PyTorch with ROCm (adjust version as needed):

   ```bash
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
   ```
7. Optional: compare with requirements file:

   ```bash
   cat python_env/requirements.txt
   ```
8. GPU utilities are already modified in NVFlare:

   ```
   NVFlare/nvflare/fuel/utils/gpu_utils.py
   ```

---

## Step 2: Download Model and Dataset

1. Set Hugging Face cache folder:

   ```bash
   export HF_HOME="./hf_home"
   export HF_HUB_DISABLE_TELEMETRY=1
   mkdir -p "$HF_HOME"
   ```
2. Create model folder and download the model:

   ```bash
   mkdir model
   python download_model.py
   ```

   * After downloading, folder should look like:

     ```
     ./model/bert-base-uncased
     ```
3. Dataset location:

   ```
   experiment_folder/data/nlp-ner
   ```

   * Each client gets its own split for train and test.

---

## Step 3: Edit SLURM File

1. Open `slurm_debug_new.slurm`.
2. On **line 27**, set `LOCATION` to your current path:

   ```bash
   pwd
   export LOCATION=<current_path>
   ```

---

## Step 4: Submit SLURM Job

1. Configure job variables in `slurm_debug_new.slurm`:

| Variable                               | Description                          |
| -------------------------------------- | ------------------------------------ |
| `#SBATCH --job-name=test_nvflare`      | Name of the job                      |
| `#SBATCH -A csc666`                    | Project/Account                      |
| `#SBATCH --qos=normal`                 | Optional, system-dependent           |
| `#SBATCH --partition=batch`            | System partition                     |
| `#SBATCH --time=02:00:00`              | Recommended for real runs            |
| `#SBATCH --output` & `--error`         | Log files                            |
| `#SBATCH --nodes`                      | Number of nodes for server + clients |
| `#SBATCH --ntasks`                     | Total tasks = clients + server       |
| `export LOG_LEVEL=INFO`                | Logging level                        |
| `export JOB_NAME=bert_ncbi_gaussian_8` | Job identifier                       |
| `export MEM_EACH_GPU=60`               | Memory per GPU (Frontier ~64GB)      |

2. Load necessary modules (Frontier example):

   ```bash
   module load PrgEnv-gnu/8.5.0
   module load miniforge3/23.11.0-0
   module load rocm/5.6.0
   module load craype-accel-amd-gfx90a
   ```
3. Hugging Face setup:

   ```bash
   export HF_HOME=$LOCATION/hf_home
   export HF_HUB_DISABLE_TELEMETRY=1
   ```
4. Server readiness check before launching clients:

   ```bash
   while [ ! -f "$SERVER_READY_FILE" ]; do
       sleep 10
   done
   ```
5. Start NVFlare admin:

   ```bash
   python start_admin.py
   ```

---

## Step 5: Multiple Node Cases (Inter-node)

| Clients | Nodes | NTASKS | JOB_NAME              | srun example                                                                                        |
| ------- | ----- | ------ | --------------------- | --------------------------------------------------------------------------------------------------- |
| 8       | 2     | 9      | bert_ncbi_gaussian_8  | `srun --ntasks=8 --nodes=1 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &`  |
| 16      | 3     | 17     | bert_ncbi_gaussian_16 | `srun --ntasks=16 --nodes=2 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &` |
| 24      | 4     | 25     | bert_ncbi_gaussian_24 | `srun --ntasks=24 --nodes=3 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &` |
| 32      | 4     | 33     | bert_ncbi_gaussian_32 | `srun --ntasks=32 --nodes=4 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &` |
| 40      | 5     | 41     | bert_ncbi_gaussian_40 | `srun --ntasks=40 --nodes=5 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &` |
| 48      | 6     | 49     | bert_ncbi_gaussian_48 | `srun --ntasks=48 --nodes=6 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &` |

---

## Step 6: Intra-node Setup (Multiple Clients on One Node)

* If SLURM allows multiple `srun` on one node, e.g., 48 clients:

  * Change:

    ```bash
    #SBATCH --nodes=1
    #SBATCH --ntasks=49
    export JOB_NAME=bert_ncbi_gaussian_48
    srun --ntasks=$NUM_CLIENTS --nodes=1 --ntasks-per-gpu=8 --gpu-bind=closest setup.sh $NAME frontier client &
    ```
* If the above does not work:

  1. Allocate node manually:

     ```bash
     salloc --nodes=1 --ntasks=49
     ```
  2. Activate Python environment:

     ```bash
     source python_env/env/bin/activate
     ```
  3. Set location variable:

     ```bash
     export LOCATION=$(pwd)
     ```
  4. Set job name and HF cache:

     ```bash
     export JOB_NAME=bert_ncbi_gaussian_8
     export HF_HOME=$LOCATION/hf_home
     export HF_HUB_DISABLE_TELEMETRY=1
     ```
  5. Run intranode setup script:

     ```bash
     ./1_setup_intranode.sh
     ```
  6. Start server:

     ```bash
     ./example_intranode/localhost/startup/start.sh
     ```
  7. Start clients:

     ```bash
     ./2_run_client.sh
     ```
  8. To kill all server and client processes:

     ```bash
     ./kill.sh
     ```

```

You can save this content as `NVFlare_PETINA_Setup.md` and view it with any Markdown viewer.  

If you want, I can also **add images and code snippets formatting** for download paths and screenshots to make it more visually readable. Do you want me to do that?
```

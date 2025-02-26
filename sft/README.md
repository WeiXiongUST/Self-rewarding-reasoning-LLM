## Installation instructions

```shell
conda create -n sft python=3.10.9
conda activate sft

## Get axolotl for general model, we use 0.6.0 in our experiments

git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn
# We use torch=2.5.1, flash-attn=2.7.0.post2 in our experiment 
pip install flash-attn==2.7.0

# If you encounter an error of axolotl: ModuleNotFoundError: No module named 'pynvml.nvml'; 'pynvml' is not a package
pip install nvidia-ml-py3
# also edit axolotl/src/axolotl/utils/bench.py (line 6) to: ``from pynvml import NVMLError''


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed==0.16.1
pip install transformers==4.48.1
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```

## Running SFT
You can update the home dir and run the following command to fine-tuning your model:

```sh
torchrun --nproc_per_node 8 --master_port 20001 -m axolotl.cli.train qwen.yaml --deepspeed YOUR_HOME_DIR/axolotl/deepspeed_configs/zero3_bf16.json
```

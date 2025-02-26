# PPO Training

We implement the PPO training by veRL package and we provide the script here. Please follow [veRL's document](https://verl.readthedocs.io/en/latest/start/install.html) to set up the training environment. However, please ensure that we install sympy==1.12.

## Data prepration of the numia prompt set

First, we need to move the numia_process.py to the verl/examples/data_preprocess/ folder. Then, run the data prepration script by:
```sh 
python verl/examples/data_preprocess/numina_process.py
```
This will prepare the training set and validation set to parquet format.

## Reward score modification

We also need to modify the current reward score used in veRL. 

First, we need to delete the original verl/utils/reward_score/\_\_init\_\_.py, then move the reward_score_init.py to the verl/utils/reward_score/ folder with the name \_\_init\_\_.py. We mainly ask the veRL to use the prime math function to compute reward for the numina dataset.

Second, we need to delete the original verl/utils/reward_score/prime_math/\_\_init\_\_.py,  then move the prime_math_init.py to the verl/utils/reward_score/prime_math/ folder with the name \_\_init\_\_.py. We mainly make some modification to ensure that the reward is computed with respect to the last answer.

## Running PPO training with numina prompt set
Similarly, now we move the verl_example.sh to the examples/ppo_trainer/ folder. Then, we need to first set up the environment by 

```sh
export VLLM_ATTENTION_BACKEND=XFORMERS
```
Otherwise, we may encounter the illegal memory error. Then, we can start the PPO training by

```
bash examples/ppo_trainer/verl.sh
```


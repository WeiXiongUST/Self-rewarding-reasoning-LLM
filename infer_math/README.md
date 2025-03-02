# Generation
This repository provides scripts and intermediate datasets for sequential rejection sampling using the Qwen2.5-Math-7B-base model. Our method involves sequentially prompting the base model to generate Chain-of-Thought (CoT) reasoning, self-rewarding signals, and revised responses in a step-by-step manner. Compared to LLaMA-based experiments, this process requires additional effort to clean the prompt and filter the data, likely because the Qwen base model is not fine-tuned for instruction-following.

## Environment setup
We first set up the environment. 
```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# You can also try other version of vllm such as 0.6.3
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.48.1
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!

pip install antlr4-python3-runtime==4.7.2
pip install sympy==1.12
pip install latex2sympy2==1.9.1
pip install word2number==1.1
```

## Sequential Rejection Sampling

**Step 1: Generate Initial Chain-of-Thought Reasoning Paths**

- Run process_prompt_turn1.py to prepare the prompts.
- Use the following script to generate initial CoT reasoning paths. This script will: (1) Start 8 separate VLLM processes for data generation; (2) Merge the generated data; and (3) Label the data using a rule-based reward function.


```sh
bash run_generation1.sh
```

**Step 2: Generate Self-Rewarding Signals**

- Run process_prompt_turn2.py to prepare the prompts.
- Update run_generation1.sh to modify the base_path and jsonl_input (the prompt set).
- Run the script to generate self-rewarding signals.


**Step 3.** Finally, you can run the process_prompt_turn3.py to prepare the prompt for generating the revised responses, and the process is similar to step 2.


## Intermediate datasets

We provide the dataset checkpoints on Huggingface:
- Example of raw dataset: [RLHFlow/self_rewarding_ift_example_raw_data1](https://huggingface.co/datasets/RLHFlow/self_rewarding_ift_example_raw_data1)
- Example of the self-rewarding IFT dataset: [RLHFlow/self_rewarding_ift_example](https://huggingface.co/datasets/RLHFlow/self_rewarding_ift_example)
- SFT prompt dataset: [RLHFlow/self_rewarding_sft_prompt](https://huggingface.co/RLHFlow)
- Turn 1 dataset with rewards: [RLHFlow/self_rewarding_turn1_with_rewards_example](https://huggingface.co/datasets/RLHFlow/self_rewarding_turn1_with_rewards_example)
- Turn 2 dataset: [RLHFlow/self_rewarding_turn2_example](https://huggingface.co/datasets/RLHFlow/self_rewarding_turn2_example)

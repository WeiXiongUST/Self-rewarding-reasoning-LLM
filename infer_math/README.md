# Generation

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

```sh
bash run_generation1.sh
```

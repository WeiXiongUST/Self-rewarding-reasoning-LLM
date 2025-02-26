# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"

export CUDA_VISIBLE_DEVICES=0
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B"
OUTPUT_DIR="Qwen/Qwen2.5-Math-7B-Eval"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR

source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"


# Base paths and settings
initial_model="Qwen/Qwen2.5-Math-7B"
base_path="./sequential_rs_rule_reward"
mkdir $base_path
iteration_prefix="Test"
best_of_k=4
my_world_size=8
NUM_GPUS=$my_world_size
iteration_name="sequential_rs"
jsonl_input="RLHFlow/self_rewarding_sft_prompt"
json_output="${base_path}/${iteration_prefix}_${iteration_name}"
model_output="${base_path}/${iteration_prefix}_${iteration_name}_reward.json"
model_path=$initial_model


# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5

    conda activate vllm
    infer_model=$2
    prompt_dir=$3
    output_dir=$4
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$i python ./generation/gen_hf.py \
            --model_name_or_path $model_path \
            --dataset_name_or_path $jsonl_input \
            --output_dir $json_output \
            --K $best_of_k \
            --temperature 1.0 \
            --local_index $i \
            --my_world_size $my_world_size &
    done
  
    wait # Ensure all inference processes finish
    
    # Merge the generated data
    python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.jsonl" --num_datasets 8
    
    # Perform reward labeling
    python reward_labeling.py --dataset_name_or_path "${output_dir}_data.jsonl" --output_dir $model_output
}




run_iteration "$iteration_name" "$model_path" "$jsonl_input" "$json_output" "$model_output"


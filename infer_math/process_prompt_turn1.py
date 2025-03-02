from datasets import load_dataset, DatasetDict

dataset1 = load_dataset("RLHFlow/self_rewarding_sft_prompt", split='train')


def pro(example):
    new_conv = []
    new_conv.append(
        { "content": 
"""You are a mathematical reasoning assistant. For each problem, follow these steps strictly:

1. Solve the problem using step-by-step reasoning and output the final answer within \\boxed{}.

Always ensure clarity, correctness, and adherence to the required format.""", 'role':'system'})
    
    new_conv.append(

        {"role": "user", "content": example['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}.'}
        
    )
    return {"prompt_messages": new_conv}

dataset1 = dataset1.map(pro)
dataset1.push_to_hub("RLHFlow/self_rewarding_sft_prompt"")


from datasets import load_dataset, Dataset, DatasetDict
ds = load_dataset("RLHFlow/self_rewarding_turn2_example", split='train')
N = 3 # we use at most N trajectories per prompt

def pro(example):
    proxy_arr = []
    for ans in example['answers']:
        if ans.count('[VERIFY]') != 1:
            proxy_arr.append(None)
        elif ans.count('[VERIFY] correct') == 1:
            proxy_arr.append(True)
        elif ans.count('[VERIFY] wrong') == 1:
            proxy_arr.append(False)
        else:
            proxy_arr.append(None)

    return {"proxy_arr": proxy_arr}

ds = ds.map(pro)


all_data_wrong = []
all_data_corr = []
for sample in ds:
    count_per_prompt_wrong = 0
    count_per_prompt_corr = 0
    for t in range(len(sample['answers'])):
        if sample['proxy_arr'][t] is None:
            continue
        if sample['proxy_arr'][t] != sample['first_reward']:
            continue
        new_conv = []
        new_conv.extend(sample['prompt_messages'])
        if not sample['first_reward']:
            if count_per_prompt_wrong >= N:
                continue
            new_conv.append(
            {"content": sample['answers'][t].replace("\\boxed{\\text{[VERIFY] wrong.}}", "[VERIFY] wrong.").replace("\\boxed{[VERIFY] wrong.}", "[VERIFY] wrong.").split("[VERIFY] wrong.")[0] + "[VERIFY] wrong.",
             "role": "assistant"})
            new_conv.append(
                { "content":"3. please identify the mistake in your previous reasoning, revise your reasoning path and output a corrected final answer within \\boxed{}.", 'role':'user'})
            all_data_wrong.append(
                {
                    'gt': sample['gt'],
                    'prompt_messages': new_conv,
                    'first_reward': sample['first_reward']
                 }
            )
            count_per_prompt_wrong += 1
        else:
            if count_per_prompt_corr >= N:
                continue
            new_conv.append(
            {"content": sample['answers'][t].replace("\\boxed{\\text{[VERIFY] correct.}}", "[VERIFY] correct.").replace("\\boxed{[VERIFY] correct.}", "[VERIFY] correct.").split("[VERIFY] correct.")[0] + "[VERIFY] correct.",
             "role": "assistant"})
            all_data_corr.append(
                {
                    'gt': sample['gt'],
                    'prompt_messages': new_conv,
                    'first_reward': sample['first_reward']
                 }
            )

keys = all_data_wrong[0].keys()  
dict_data = {key: [d[key] for d in all_data_wrong] for key in keys}
dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub('YOUR DIR1')


keys = all_data_corr[0].keys()  
dict_data = {key: [d[key] for d in all_data_corr] for key in keys}
dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub('YOUR DIR2')


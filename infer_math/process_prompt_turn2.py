from datasets import load_dataset, DatasetDict, Dataset

ds = load_dataset("RLHFlow/self_rewarding_turn1_with_rewards_example", split='train')

all_data = []
for sample in ds:
    t = 0
    for ans in sample['answers']:
        if 'boxed' not in ans:
            t += 1
            continue
        new_conv = []
        new_conv.extend(sample['prompt_messages'])
        new_conv.append(
            {"content": ans,
             "role": "assistant"}
        )
        new_conv.append(
        { "content": 
"""2. Perform a self-evaluation:
   - You may include reasoning to verify correctness.
   - However, your final self-evaluation **must** be in one of the following formats:
     ```
     [VERIFY] correct.
     ```
     or  
     ```
     [VERIFY] wrong.
     ```""", 'role':'user'})
        
        all_data.append(
            {
                "prompt_messages": new_conv,
                'gt': sample['gt'],
                'first_reward': sample['rewards'][t]
            }
        )


keys = all_data[0].keys()  

dict_data = {key: [d[key] for d in all_data] for key in keys}


dataset = Dataset.from_dict(dict_data)

DatasetDict({'train': dataset}).push_to_hub('YOUR DIR')




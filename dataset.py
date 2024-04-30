import torch
from torch.utils.data import Dataset

class ARQMathDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        q = self.df['q'][index]
        a = self.df['a'][index]
        
        inputs = self.tokenizer(
            q,
            a,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation='longest_first',
            pad_to_max_length=True,
            return_tensors='pt',
            return_token_type_ids=True
        )

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'token_type_ids': inputs['token_type_ids'][0],
            'labels': torch.tensor(self.df['rel'][index], dtype=torch.long)
        }
    
    def __len__(self):
        return self.len
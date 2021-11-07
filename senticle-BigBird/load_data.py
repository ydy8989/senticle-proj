import torch

def tokenized_dataset(dataset, tokenizer, max_seqlen):

    tokenized_sentences = tokenizer(
        list(dataset),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seqlen,
        add_special_tokens=True,
    )
    return tokenized_sentences

class ZumDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from neuronx_distributed_training.lightning_modules.data.utils import pad_token_list, convert_dict_to_list, pad_token_tensor
import torch
from torch.utils.data import Dataset


class PaddedDataset(Dataset):
    def __init__(self, dataset, pad_id_map, pad_size=4096):
        self.dataset = dataset
        self.pad_size = pad_size
        self.pad_id_map = pad_id_map

        self.samples = self.pad_datasets(dataset, pad_size, pad_id_map)

    def pad_datasets(self, data, pad_size, pad_id_map):
        """Pad the record upto the pad size so as to have constant length for all records

        Args:
            data (Dataset): Input tokenized data. {'input_ids':[[],[]...], 'attention_mask'=[[],[]...],'labels'=[[],[],...]}
            pad_size (int): packing size
            pad_id_map (dict): map of padding tokens for all dicts in dataset

        Returns:
            List[dicts]: _description_ like [{},{},{}]
        """

        pad_data = {}
        for k in ['input_ids','attention_mask','labels']:
            pad_data[k] = pad_token_list(data[k], pad_size, pad_id_map[k])

        return convert_dict_to_list(pad_data)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class PaddedDPODataset(Dataset):

    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __init__(self, dataset, pad_size=4096):
        self.dataset = dataset
        self.pad_size = pad_size


        self.samples = self.pad_datasets(dataset, pad_size)

    def pad_datasets(self, data, pad_size):

        padded_batch = {}

        keys_to_pad = [k for k in data[0].keys() if k.endswith(("_input_ids", "_attention_mask", "_labels"))]

        for key in keys_to_pad:
            padding_value = None
            if key.endswith("_input_ids"):
                if self.pad_token_id is None:
                    raise ValueError(
                        "Padding is enabled, but the tokenizer is not configured with a padding token."
                        " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                        " before calling the trainer."
                    )
                padding_value = self.pad_token_id
            elif key.endswith("_labels"):
                padding_value = self.label_pad_token_id
            elif key.endswith("_attention_mask"):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{key}'")

             # Set padding side based on the key
            if key in ["prompt_input_ids", "prompt_attention_mask"]:
                padding_side = "left"
            else:
                padding_side = "right"

            dtype = torch.int64
            to_pad = [torch.tensor(ex[key], dtype=dtype) for ex in data]
            padded_batch[key] = pad_token_tensor(to_pad, padding_size=self.pad_size, padding_value=padding_value, padding_side=padding_side)

        # Handle other keys that don't need padding (refrence_probs passed in batch are handled here)
        for k in data[0].keys():
            if k not in keys_to_pad:
                padded_batch[k] = [ex[k] for ex in data]

        # Converts dict of key and value to lists with dicts.
        samples = []
        for idx in range(len(padded_batch['chosen_input_ids'])):
            samples.append({k:padded_batch[k][idx] for k in padded_batch.keys()})

        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

from neuronx_distributed_training.lightning_modules.data.utils import pad_token_list,convert_dict_to_list

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

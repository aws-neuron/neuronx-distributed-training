# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import Dataset
from neuronx_distributed_training.lightning_modules.data.utils import pad_tokens,convert_dict_to_list

class ConcatDataset(Dataset):

    def __init__(self, dataset, EOS_ID, pad_id_map, chunk_size=4096):
        """ Construct the dataset samples using the datasets.dataset object

        Args:
            dataset (Dataset): Input tokenized data
            EOS_ID (int): end of sentence ID
            pad_id_map (dict): map of padding tokens for all dicts in dataset
            chunk_size (int, optional): packing size. Defaults to 4096.
        """
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.eos = EOS_ID
        self.pad_id_map = pad_id_map    
        self.samples = self.pack_datasets(dataset, chunk_size, EOS_ID, pad_id_map)          
    
    def pack_datasets(self, data, chunk_size, EOS_ID, pad_id_map):
        """Pack multiple records into one single record of chunk size

        Args:
            data (Dataset): Input tokenized data {'input_ids':[[],[]...], 'attention_mask'=[[],[]...],'labels'=[[],[],...]}
            chunk_size (int): packing size
            EOS_ID (int): end of sentence ID
            pad_id_map (dict): map of padding tokens for all dicts in dataset

        Returns:
            List[dicts]: List of dicts where each of the dict is one packed record. And each packed record can have more than 1 original record
        """
        # read all the data, shuffle it, pack it till max seq len of input+output.
        packed_data = {'input_ids':[], 'attention_mask':[], 'labels':[]}                    

        idx = 0
        len_data_points = len(data['input_ids'])

        # Need to read the dataset columns before doing the seq2seq packing otherwise it will not be performant
        # Once coiped over its type is list instead of dataset 
        original_tokenized_data = { 'input_ids':data['input_ids'],
                                    'labels': data['labels'],
                                    'attention_mask': data['attention_mask']}        
        
        original_pad_id = pad_id_map['input_ids']
        
        # Another optimization to speed up below packing is to have workers do the data packing on subset of data
        while idx<len_data_points:
            temp_packed_data = {'input_ids':[], 'labels':[], 'attention_mask':[]}
            pad_id_map['input_ids'] = EOS_ID

            while idx<len_data_points and \
                len(temp_packed_data['input_ids'])+len(original_tokenized_data['input_ids'][idx]) < chunk_size:
            
                for k in ['input_ids','attention_mask','labels']:
                    temp_packed_data[k].extend(original_tokenized_data['input_ids'][idx] + [pad_id_map[k]])

                idx += 1

            # if any record has more than chunk size tokens ignore it.
            if idx<len_data_points and len(original_tokenized_data['input_ids'][idx])>chunk_size-1:
                idx+=1

            pad_id_map['input_ids'] = original_pad_id
            if temp_packed_data['input_ids']:
                for k in ['input_ids','attention_mask','labels']:            
                    packed_data[k].append(pad_tokens(temp_packed_data[k], chunk_size, pad_id_map[k]))                                    
                    
        # packed_data is a dict of keys inputs_ids and labels and attention_mask, each value has multiple records
        # We need list of dicts instead of dict values with lists.        
       
        return convert_dict_to_list(packed_data)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from neuronx_distributed.parallel_layers import parallel_state
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
import pyarrow as pa
import transformers
import torch
from transformers import AutoTokenizer
from functools import partial
import collections
import logging

from neuronx_distributed_training.lightning_modules.data.datasets.ConcatDataset import ConcatDataset
from neuronx_distributed_training.lightning_modules.data.datasets.PaddedDataset import PaddedDataset
from neuronx_distributed_training.lightning_modules.data.base import BaseDataModule

logger = logging.getLogger(__name__)


class SFTDataModule(BaseDataModule):
    """SFT data class that loads pretrained tokenizer, tokenizes, packs and has the dataloader along with distributed sampler

    Args:
        BaseDataModule : Base data class
    """
    SRC_NAME = "input"
    DST_NAME = "output"

    def setup(self, stage=None):
        super().setup()

        self.data_sets = {
            "train": None,  # Placeholder for training dataset
            "val": None   # Placeholder for validation dataset
        }
        dataset_keys = list(self.data_sets.keys())

        self.tokenizer = self.load_tokenizer() 
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_type_id
        self.tokenizer.padding_side = 'right'

        self._build_vocab()

        self.load_datasets(dataset_keys) # updated the self.data_sets
        
        self.prompt_datasets(dataset_keys) # apply prompts for the dataset
        
        self.tokenize_datasets(dataset_keys) # tokenize all the dict keys if they are not none and replace the values with tokenized datasets
        
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.data.tokenizer.type)
    
    def load_datasets(self, data_split_list):
        """Load the dataset of arrow and jsonl format. Loads to class instance attribute directly and doesnt return anything

        Args:
            data_split_list (dict): All dataset splits need to be mentioned here 

        Raises:
            ValueError: If input specified is other than arrow or jsonl
        """

        for partition in data_split_list:            
            path = getattr(self.config.data, partition + "_dir") # failing fix this on 30th
            if not path:
                logger.warn(f"Path is empty for {partition=}")
                continue

            if path.lower().endswith("jsonl") or path.lower().endswith("json"):
                logger.info(f"Loading jsonl data for {partition=}")
                self.data_sets[partition] = load_dataset("json", data_files=path, split="train")
            elif path.lower().endswith("arrow"): # arrow format
                logger.info(f"Loading arrow data for {partition=}")
                mmap = pa.memory_map(self.training_dataset_path)
                self.data_sets[partition] = pa.ipc.open_stream(mmap).read_all()            
            else:
                raise ValueError(f"Path should have .jsonl/.json/.arrow, but got {path=}")
    
    def prompt_datasets(self, data_split_list):
        """Applies templates to the input columns of the dataset

        Args:
            data_split_list (dict): All dataset splits need to be mentioned here 

        Returns:
            dataset: Returns prompted dataset
        """
        # Only apply prompts from the bigscience for now, can be extended to custom prompts easily by providing jinja templates here

        def _doc_to_prompted(document, prompt=None):
            if prompt:
                return prompt.encode(document)
            return document # any f string format can also be used here
        
        prompt = None
        if getattr(self.config.data,'dataset_name',None) and getattr(self.config.data,'prompt_name',None) :
            from promptsource.templates import DatasetTemplates # promptsource installation has issues with python versions and other dependencies.
            dataset_prompts = DatasetTemplates(self.config.data.dataset_name, self.config.data.subset_name if getattr(self.config.data,'subset_name') else None)
            prompt = dataset_prompts[getattr(self.config.data,'prompt_name')]
        
        if not prompt:
            return
        
        for partition in data_split_list:
            add_prompts = partial(_doc_to_prompted, prompt = prompt)
            self.data_sets[partition] = self.data_sets[partition].map(add_prompts, batched=False, num_proc=8) # change the worker count
        

    def tokenize_datasets(self, data_split_list):
        """Converts the text to tokens, updates the class instance attribute

        Args:
            data_split_list (dict): All dataset splits need to be mentioned here 
        """
        def _tokenize_and_modify_for_FT(document):
            src_tokens_batch = self.tokenizer(document[SFTDataModule.SRC_NAME], max_length=self.config.data.seq_length, add_special_tokens=False) # document[SRC_NAME] will have batch text lists
            dst_tokens_batch = self.tokenizer(document[SFTDataModule.DST_NAME], max_length=self.config.data.seq_length, add_special_tokens=False)

            src_input_ids = [[self.tokenizer.bos_token_id] + i_d for i_d in src_tokens_batch["input_ids"]]
            dst_input_ids = [o_d + [self.tokenizer.eos_token_id] for o_d in dst_tokens_batch["input_ids"]]
            
            return {
                "input_ids": [i_d + o_d for i_d, o_d in zip(src_input_ids, dst_input_ids)],
                "attention_mask": [[1] * (len(i_d) + len(o_d)) for i_d, o_d in zip(src_input_ids, dst_input_ids)], 
                "labels" : [[-100] * len(i_d) + o_d for i_d, o_d in zip(src_input_ids, dst_input_ids)]
            }

        for partition in data_split_list:
            if self.config.data.dev_choose_samples:
                self.data_sets[partition] = self.data_sets[partition].select(range(self.config.data.dev_choose_samples))
            self.data_sets[partition] = self.data_sets[partition].map(_tokenize_and_modify_for_FT, batched=True) 
    

    def _build_dataloader(self, data):
        """Builds dataloader for dataset, Also does the padding and packing. Run every epoch so it can shuffle for every epoch

        Args:
            data (dataset): actual dataset object that can be used with dataloader directly
        """           
        pad_id_map = {'input_ids':self.tokenizer.pad_token_id,
                      'labels':-100,
                      'attention_mask':0}

        if getattr(self.config.data,"packing",None):            
            data = ConcatDataset(data, self.tokenizer.eos_token_id, pad_id_map, self.config.data.seq_length) # packs dataset and returns new dataset
        else:
            data = PaddedDataset(data, pad_id_map, self.config.data.seq_length) # collator pads for batch length same, but we want all batches to be of same length, so choosing max seq as default

        sampler = DistributedSampler(
            data,
            num_replicas=int(self.dp_size),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            drop_last=True,
        )
        return DataLoader(
            data,
            collate_fn=transformers.DataCollatorForSeq2Seq(self.tokenizer),
            sampler=sampler,
            batch_size=int(self.config.data.global_batch_size / self.dp_size),
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )       
        
    def train_dataloader(self):
        """
        Called once per epoch only, so we can shuffle here for packing and repack here every epoch. 
        Packing can be for every epoch or per dataset, depending on user choice as randomization still happens at model input level if not at per seq level.
        """       
        return self._build_dataloader(self.data_sets['train'])

    def val_dataloader(self):
        return self._build_dataloader(self.data_sets['val'])
    
    def get_batch_length(self, batch):
        return len(batch["input_ids"])
    
    def process_global_batch(self, global_batch, global_batch_size=None):
        """Prepares the global batch for apex fwd/bwd functions.
        Global batch is a list of micro batches.
        """
        input_ids, attention_mask, labels = global_batch.values()
        loss_mask = torch.ones(labels.size(), dtype=torch.float)
        # set the lables with -100 as 0.0            
        loss_mask[labels == -100] = 0.0            
        seq_length = input_ids.numel()
        position_ids = torch.arange(seq_length, dtype=torch.int64).repeat(input_ids.shape[0], 1)
        return [
            input_ids,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
        ]

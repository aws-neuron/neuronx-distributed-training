# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# HAVE TO patch only _build_index_mappings function

"""GPT style dataset."""

import os
import time

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import (
    BlendableDataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    GPTDataset as GPTDatasetNeMo,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    _build_doc_idx,
    _build_sample_idx,
    _build_shuffle_idx,
    _num_epochs,
    _num_tokens,
    get_indexed_dataset_,
)
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import (
    deallocate_indexed_dataset_memory,
)
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import (
    make_dataset as make_indexed_dataset,
)
from nemo.core import Dataset
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig


def build_dataset(cfg, trainer, data_prefix, data_impl, num_samples, seq_length, seed, skip_warmup, tokenizer, name):
    def _build_dataset(current_data_prefix, current_num_samples):
        indexed_dataset = get_indexed_dataset_(current_data_prefix, data_impl, skip_warmup)
        total_num_of_documents = indexed_dataset.sizes.shape[0]
        # Print stats about the splits.
        logging.info(" > dataset split:")
        logging.info("     Total {} documents is : {} ".format(name, total_num_of_documents))
        drop_last = True
        if name == "valid":
            drop_last = cfg.data.get("validation_drop_last", True)
        dataset = GPTDataset(
            cfg,
            trainer,
            tokenizer,
            name,
            current_data_prefix,
            np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            indexed_dataset,
            current_num_samples,
            seq_length,
            seed,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)

    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(prefixes)):
            dataset = _build_dataset(prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets, weights, num_samples)


def build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    if isinstance(data_prefix, DictConfig):
        assert (
            data_prefix.get("train") is not None
            and data_prefix.get("test") is not None
            and data_prefix.get("validation") is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.data.splits_string is not None:
            logging.warning(cfg.data.splits_string + " ignored since data prefix is of type dictionary.")
        train_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["train"],
            data_impl,
            int(train_valid_test_num_samples[0]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "train",
        )
        validation_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["validation"],
            data_impl,
            int(train_valid_test_num_samples[1]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "valid",
        )
        test_ds = build_dataset(
            cfg,
            trainer,
            data_prefix["test"],
            data_impl,
            int(train_valid_test_num_samples[2]),
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            "test",
        )
        return train_ds, validation_ds, test_ds

    else:
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cfg,
                trainer,
                data_prefix[0],
                data_impl,
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cfg,
                trainer,
                prefixes[i],
                data_impl,
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
                tokenizer,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_n)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_n)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    tokenizer,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(" > dataset split:")

    def print_split_stats(name, index):
        logging.info("    {}:".format(name))
        logging.info(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "valid":
                drop_last = cfg.data.get("validation_drop_last", True)
            dataset = GPTDataset(
                cfg,
                trainer,
                tokenizer,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


class GPTDataset(GPTDatasetNeMo):
    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        drop_last=True,
    ):
        Dataset.__init__(self)
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.drop_last = drop_last
        self.seq_length = seq_length

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.reset_position_ids = cfg.data.get("reset_position_ids", False)
        self.reset_attention_mask = cfg.data.get("reset_attention_mask", False)
        self.eod_mask_loss = cfg.data.get("eod_mask_loss", False)
        self.eos_id = tokenizer.eos_id
        self.no_seqlen_plus_one_input_tokens = cfg.data.get("no_seqlen_plus_one_input_tokens", False)
        self.add_extra_token = 1
        if self.no_seqlen_plus_one_input_tokens:
            self.add_extra_token = 0

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get("index_mapping_dir", None)

        # create index_mapping_dir on rank 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)
            torch.distributed.barrier()

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            num_samples,
            seq_length,
            seed,
            index_mapping_dir=self.index_mapping_dir,
            drop_last=drop_last,
            add_extra_token=self.add_extra_token,
        )
        deallocate_indexed_dataset_memory(self.indexed_dataset)

    def __getitem__(self, idx):
        text = torch.from_numpy(self._get_text(idx))
        if self.add_extra_token:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = -1
        attention_mask, loss_mask, position_ids = _create_ltor_masks_and_position_ids(
            tokens,
            self.eos_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
        )
        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        # Negative index comes when we pad the last batch in MegatronPretrainingBatchSampler
        # We make the loss_mask zero to mask out loss from these samples
        if idx == -1:
            logging.info("WARNING: Got -1 as item index. Masking loss from this sample")
            loss_mask = torch.zeros_like(loss_mask)

        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


@torch.no_grad()
def _create_ltor_masks_and_position_ids(
    tokens: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
):
    """Create `attention_mask`, `loss_mask`, and `position_ids`.

    This function is modified :func:`get_ltor_masks_and_position_ids` in nemo/collections/nlp/modules/common/megatron/utils.py:
    `get_ltor_masks_and_position_ids` assumes a microbatch of ``tokens``, i.e. 2D tensor while
    this function assumes ``tokens`` to be 1D tensor.

    Args:
        tokens: A 1D tensor that holds the indices of tokens.
        eod_token:
        reset_position_ids:
        reset_attention_mask:
        eod_mask_loss

    """
    assert tokens.ndim == 1
    seq_length = tokens.numel()
    loss_mask = torch.ones(seq_length, dtype=torch.float)
    if eod_mask_loss:
        loss_mask[tokens == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.int64)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids:
        # Find indices where EOD token is.
        eod_index = position_ids[tokens == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1
    assert not reset_attention_mask, "Neuron flow does not support attention mask reset"
    attention_mask = torch.tensor([True])
    # Needs to be a dummy tensor since a whole bunch of batch things are done under the hood.
    # Size does not matter, it will be replaced by device tensor
    return attention_mask, loss_mask, position_ids


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    index_mapping_dir: str = None,
    drop_last: bool = True,
    add_extra_token: int = 1,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    if index_mapping_dir is not None:
        _filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            logging.info(" > WARNING: could not find index map files, building " "the indices on rank 0 ...")

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(" > only one epoch required, setting " "separate_last_epoch to False", flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - add_extra_token
                ) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, "last epoch number of samples should be non-negative."
                num_samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
                assert last_epoch_num_samples < (
                    num_samples_per_epoch + 1
                ), "last epoch number of samples exceeded max value."
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
                if separate_last_epoch:
                    string = (
                        " > last epoch number of samples ({}) is smaller "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to True"
                    )
                else:
                    string = (
                        " > last epoch number of samples ({}) is larger "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to False"
                    )
                print(string.format(last_epoch_num_samples, num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            logging.info(
                " > elasped time to build and save doc-idx mapping " "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            try:
                from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
                    compile_helper,
                )

                compile_helper()
                from nemo.collections.nlp.data.language_modeling.megatron import helpers
            except ImportError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )

            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last, add_extra_token
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                              num_epochs, tokens_per_epoch, drop_last, add_extra_token)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            logging.info(
                " > elasped time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            logging.info(
                " > elasped time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )
    import torch_xla.core.xla_model as xm

    xm.rendezvous("dataset")

    torch.distributed.barrier()

    # Load mappings.
    start_time = time.time()
    logging.info(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    logging.info(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    logging.info(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    logging.info("    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
    logging.info("    total number of samples: {}".format(sample_idx.shape[0]))
    logging.info("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx

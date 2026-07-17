# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "src" / "neuronx_distributed_training" / "lightning_modules" / "data"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_concat_dataset():
    with patch.dict(sys.modules):
        load_module(
            "neuronx_distributed_training.lightning_modules.data.utils",
            DATA_ROOT / "utils.py",
        )
        module = load_module(
            "concat_dataset_under_test",
            DATA_ROOT / "datasets" / "ConcatDataset.py",
        )

    return module.ConcatDataset


ConcatDataset = load_concat_dataset()


class TestConcatDataset(unittest.TestCase):
    def test_pack_datasets_reads_each_source_field(self):
        data = {
            "input_ids": [[10, 11]],
            "attention_mask": [[1, 1]],
            "labels": [[-100, 11]],
        }
        pad_id_map = {"input_ids": 0, "attention_mask": 0, "labels": -100}

        dataset = ConcatDataset(data, EOS_ID=99, pad_id_map=pad_id_map, chunk_size=4)

        self.assertEqual(
            dataset.samples,
            [
                {
                    "input_ids": [10, 11, 99, 0],
                    "attention_mask": [1, 1, 0, 0],
                    "labels": [-100, 11, -100, -100],
                }
            ],
        )

    def test_pack_datasets_preserves_exact_length_record(self):
        data = {
            "input_ids": [[10, 11, 12, 13]],
            "attention_mask": [[1, 1, 1, 1]],
            "labels": [[-100, -100, 12, 13]],
        }
        pad_id_map = {"input_ids": 0, "attention_mask": 0, "labels": -100}

        dataset = ConcatDataset(data, EOS_ID=99, pad_id_map=pad_id_map, chunk_size=4)

        self.assertEqual(
            dataset.samples,
            [
                {
                    "input_ids": [10, 11, 12, 13],
                    "attention_mask": [1, 1, 1, 1],
                    "labels": [-100, -100, 12, 13],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()

## Installation:

#### Build apex==0.1 wheel
1. Clone apex repo

```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 25.07
```

2. Replace the contents of the setup.py with the following contents:

```
import sys
import warnings
import os
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, load

setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
    ),
    install_requires=["packaging>20.6",],
    description="PyTorch Extensions written by NVIDIA",
)
```

3. Build the wheel using the command:

```
python setup.py bdist_wheel
```

### TRN1 PC - DO-NOT-DELETE-NEURON-TRAINING-TRN1-PC
- ARN: `arn:aws:cloudformation:us-west-2:621547421844:stack/DO-NOT-DELETE-NEURON-TRAINING-PC/6426e3a0-26e2-11f0-963d-0a0ac6f2872f`
- HeadNode `i-07bcd998d1eb9027a` IP: `ec2-44-234-60-97.us-west-2.compute.amazonaws.com`
- FSX: File System ID: `fs-074e75aa92df10ac1`

#### Install the neuron deps:

Install the neuron packages using the command:

```
pip install --upgrade neuronx-cc==2.* torch-neuronx torchvision --extra-index-url https://pip.repos.neuron.amazonaws.com
```

#### Install requirements and neuronx_distributed packages

```
pip install -r requirements.txt apex/dist/apex-0.1-py3-none-any.whl
pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com
```

After installing the requirements, we need to patch some of the installations so run

```
chmod +x install_setup.sh
./install_setup.sh
```

#### Install NeuronxDistributedTraining (NxDT)

```
pip install neuronx_distributed_training --extra-index-url https://pip.repos.neuron.amazonaws.com
```

## How to run.

#### Setup dataset:

Run the following steps to download the dataset:

```
wget wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/config.json ~/
wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/get_dataset.py
```

#### Download/Tokenize the dataset:

Run the following command to tokenize the dateset:

```
python get_dataset.py --llama-version 3
```

If you are working with llama2, then change the version to 2. Note: for the above command to work,
you need to download the tokenizer using the following snippet:

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', token='your_own_hugging_face_token')
# For llama2 uncomment line below
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token='your_own_hugging_face_token')

tokenizer.save_pretrained(".")
```

After running the get_dataset.py, you should see the dataset been downloaded and tokenized at `~/example_datasets/`


#### Run the example

Clone the examples folder to trn1 instance. Before running training, ensure the config path and dataset path are 
correctly set inside `conf/hf_llama_7B_config.yaml`.

Once configured, you can then run the parallel_compile using:

```
COMPILE=1 CONF_FILE=hf_llama3_8B_config ./train.sh
```

This should extract all the graphs and compile them in parallel. Once done, you can then run  the training job using:

```
CONF_FILE=hf_llama3_8B_config ./train.sh
```


## Contributing

#### Formatting code

To format the code, use the following command:

```
pre-commit run --all-files
```

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

def pad_token_list(tokenList, pad_size, pad_token):
    '''
    tokenList is [[12,34,....],[2343,565,55,...]..] List of lists where internal list is for each record tokenized (record can be a packed record or unpacked)
    '''
    return [each_record + [pad_token] * (pad_size-(len(each_record)%pad_size)) for each_record in tokenList]

def pad_tokens(tokens, pad_size, pad_token):
    '''
    tokens is 1D list of tokens and output is 1D list with appended padding
    '''
    if len(tokens) == pad_size:
        return tokens
    return tokens + [pad_token] * (pad_size-(len(tokens)%pad_size))

def convert_dict_to_list(dict_data):
    """
    Converts dict of key and value of lists of list to lists with dicts.
    """
    samples = []
    for idx in range(len(dict_data['input_ids'])):
        samples.append({k:dict_data[k][idx] for k in dict_data.keys()})
    return samples

def pad_token_tensor(tensors, padding_size=4096, padding_value= 0, padding_side="right"):
    """
    Pads a list of tensors to a fixed size based on the padding side

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    """
    # Determine the maximum shape for each dimension
    output_shape = [padding_size]

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

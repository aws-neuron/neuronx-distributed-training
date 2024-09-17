# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
        
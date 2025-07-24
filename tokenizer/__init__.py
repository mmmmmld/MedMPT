# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/31 13:04

import warnings

tokenizer_dict = {
    'clinicalbert': '../../models/clinicalbert',
    'chinesebert': '../../models/chinesebert',
}

def verify_special_tokens(config, tokenizer):
    default_vocab_size = tokenizer.vocab_size
    default_unk_index = tokenizer.unk_token_id
    default_pad_index = tokenizer.pad_token_id
    default_mask_index = tokenizer.mask_token_id
    try:
        default_sos_index = tokenizer.sos_token_id
        default_eos_index = tokenizer.eos_token_id
    except:
        default_sos_index = tokenizer.cls_token_id
        default_eos_index = tokenizer.sep_token_id

    set_vocab_size = config.tokenizer.vocab_size
    set_unk_index = config.tokenizer.unk_index
    set_pad_index = config.tokenizer.pad_index
    set_mask_index = config.tokenizer.mask_index
    set_sos_index = config.tokenizer.sos_index
    set_eos_index = config.tokenizer.eos_index

    if default_vocab_size != set_vocab_size:
        warnings.warn('[VOCAB_SIZE] set in configuration is different from defaults!', UserWarning)
    if default_unk_index != set_unk_index:
        warnings.warn('[UNK] index set in configuration is different from defaults!', UserWarning)
    if default_pad_index != set_pad_index:
        warnings.warn('[PAD] index set in configuration is different from defaults!', UserWarning)
    if default_mask_index != set_mask_index:
        warnings.warn('[MASK] index set in configuration is different from defaults!', UserWarning)
    if default_sos_index != set_sos_index:
        warnings.warn('[SOS] index set in configuration is different from defaults!', UserWarning)
    if default_eos_index != set_eos_index:
        warnings.warn('[EOS] index set in configuration is different from defaults!', UserWarning)

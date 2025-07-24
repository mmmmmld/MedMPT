# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/04/26 11:37


import torch
from transformers import BertConfig, BertModel, BertLMHeadModel

path_dict = {
    "bert": "/root/maliangdi/models/bert",
    "chinesebert": "/root/maliangdi/models/chinesebert",
}

class TransformerEncoder(torch.nn.Module):
    def __init__(self, backbone="chinesebert", num_hidden_layers=6, output_hidden_states=True, output_attentions=True,
                 pretrained=True, freeze_pretrained=False, freeze_pretrained_layers=[], **kwargs):
        super().__init__()
        config = BertConfig.from_pretrained(path_dict[backbone])
        config.num_hidden_layers = num_hidden_layers
        config.output_hidden_states = output_hidden_states
        config.output_attentions = output_attentions
        if pretrained:
            self.bert = BertModel.from_pretrained(path_dict[backbone], config=config)
            print("load BertModel from pretrained")
        else:
            self.bert = BertModel(config=config)
        self.config = config  # hidden_size = config.hidden_size

        if freeze_pretrained:
            if len(freeze_pretrained_layers) > 0:
                if 'embeddings' in freeze_pretrained_layers:
                    for param in list(self.bert.embeddings.parameters()):
                        param.requires_grad = False
                    freeze_pretrained_layers.remove('embeddings')
                for layer_idx in freeze_pretrained_layers:
                    for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                        param.requires_grad = False
            else:
                for param in self.bert.parameters():
                    param.requires_grad = False

    def forward(self, text, return_dict=True):
        out = self.bert(**text, return_dict=return_dict)
        return out


class ImageCaptioningTransformer(torch.nn.Module):
    def __init__(self, backbone="chinesebert", vision_width=768, vocab_size=21128, num_hidden_layers=6,
                 output_hidden_states=True, output_attentions=True, pretrained=True, freeze_pretrained=False,
                 freeze_pretrained_layers=[], **kwargs):
        super(ImageCaptioningTransformer, self).__init__()

        # 使用预训练 BERT 模型的配置文件并修改层数
        config = BertConfig.from_pretrained(path_dict[backbone])
        config.vocab_size = vocab_size
        config.num_hidden_layers = num_hidden_layers
        config.encoder_width = vision_width
        config.output_hidden_states = output_hidden_states
        config.output_attentions = output_attentions
        config.is_decoder = True
        config.add_cross_attention = True

        # 用预训练模型的参数初始化 BERT 模型
        if pretrained:
            self.bert = BertLMHeadModel.from_pretrained(path_dict[backbone], config=config)
            print("load BertLMHeadModel from pretrained")
        else:
            self.bert = BertLMHeadModel(config=config)
        self.config = config  # hidden_size = config.hidden_size

    def forward(self, text, image_embed, image_atts=None, return_dict=True, output_attn=False):
        if image_atts is None:
            image_atts = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(image_embed.device)
        out = self.bert(**text,
                        encoder_hidden_states=image_embed,
                        encoder_attention_mask=image_atts,
                        return_dict=return_dict)
        if output_attn:
            return out.logits, out.cross_attentions

        return out.logits  # (bs, max_len, voc_size)





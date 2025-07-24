# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
from torch import nn
from typing import Callable, Dict, List, Tuple
import warnings
import functools
from functools import partial
import torch.nn.functional as F
import transformers
transformers.logging.set_verbosity_error()
from .modules.fusion_modules import FusionTransformer
from .modules.bert import TransformerEncoder, ImageCaptioningTransformer
from .modules.mae import MAEViTEncoder, MAEViTDecoder
from .modules import pooler
from utils import link


class PretrainModel(nn.Module):
    def __init__(self, args, **kwargs):
        super(PretrainModel, self).__init__()
        # define parameters
        self.args = args
        self.mask_ratio_q = self.args.vision_mask_ratio_q
        self.mask_ratio_k = self.args.vision_mask_ratio_k
        # self.global_feature_size = self.args.global_feature_size
        self.queue_size_slice = self.args.queue_size_slice
        self.queue_size_scan = self.args.queue_size_scan
        self.momentum = self.args.momentum
        self.temp_image = nn.Parameter(self.args.temperature_image * torch.ones([]))
        self.temp_multimodal = nn.Parameter(self.args.temperature_multimodal * torch.ones([]))

        # create vision encoder and momentum vision encoder - vit-b/16
        self.vision_encoder_q = MAEViTEncoder(
            patch_size=self.args.vision_patch_size, embed_dim=self.args.vision_embed_dim,
            depth=self.args.vision_depth, num_heads=self.args.vision_num_heads,
            decoder_embed_dim=self.args.vision_decoder_embed_dim, decoder_depth=self.args.vision_decoder_depth,
            decoder_num_heads=self.args.vision_decoder_num_heads, mlp_ratio=self.args.vision_mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=self.args.vision_pretrained,
            checkpoint=self.args.vision_pretrained_weight,
            freeze_pretrained_layers=self.args.vision_freeze_pretrained_layers)
        self.vision_width = self.vision_encoder_q.vision_width
        self.vision_fusion_q = FusionTransformer(embed_dim=self.vision_width)
        self.vision_proj_slice_q = nn.Sequential(
            nn.Linear(self.vision_width, self.vision_width),
            nn.ReLU(),
            nn.Linear(self.vision_width, self.vision_width),
        )
        self.vision_proj_scan_q = nn.Sequential(
            nn.Linear(self.vision_width, self.vision_width),
            nn.ReLU(),
            nn.Linear(self.vision_width, self.vision_width),
        )
        self.vision_encoder_k = MAEViTEncoder(
            patch_size=self.args.vision_patch_size, embed_dim=self.args.vision_embed_dim,
            depth=self.args.vision_depth, num_heads=self.args.vision_num_heads,
            decoder_embed_dim=self.args.vision_decoder_embed_dim, decoder_depth=self.args.vision_decoder_depth,
            decoder_num_heads=self.args.vision_decoder_num_heads, mlp_ratio=self.args.vision_mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=self.args.vision_pretrained,
            checkpoint=self.args.vision_pretrained_weight,
            freeze_pretrained_layers=self.args.vision_freeze_pretrained_layers)
        self.vision_fusion_k = FusionTransformer(embed_dim=self.vision_width)
        self.vision_proj_slice_k = nn.Sequential(
            nn.Linear(self.vision_width, self.vision_width),
            nn.ReLU(),
            nn.Linear(self.vision_width, self.vision_width),
        )
        self.vision_proj_scan_k = nn.Sequential(
            nn.Linear(self.vision_width, self.vision_width),
            nn.ReLU(),
            nn.Linear(self.vision_width, self.vision_width),
        )

        # create text encoder
        self.text_encoder_q = TransformerEncoder(
            backbone=self.args.text_backbone,
            num_hidden_layers=self.args.text_num_hidden_layers,
            output_hidden_states=self.args.text_output_hidden_states,
            output_attentions=self.args.text_output_attentions,
            freeze_pretrained=self.args.text_freeze_pretrained,
            freeze_pretrained_layers=self.args.text_freeze_pretrained_layers
        )
        self.text_width = self.text_encoder_q.config.hidden_size
        self.text_proj_q = nn.Sequential(
            nn.Linear(self.text_width, self.text_width),
            nn.ReLU(),
            nn.Linear(self.text_width, self.text_width),
        )
        self.att_mean_pooler = pooler.MeanPooling()  # pooling token embeddings as global sentence feature

        self.model_pairs = [[self.vision_encoder_q, self.vision_encoder_k],
                            [self.vision_fusion_q, self.vision_fusion_k],
                            [self.vision_proj_slice_q, self.vision_proj_slice_k],
                            [self.vision_proj_scan_q, self.vision_proj_scan_k],
                            ]
        self.copy_params()

        self.vision_to_text = nn.Linear(self.vision_width, self.text_width)

        # create the queue
        self.register_buffer("image_queue_slice", torch.randn(self.vision_width, self.queue_size_slice))
        self.register_buffer("image_queue_scan", torch.randn(self.vision_width, self.queue_size_scan))
        self.register_buffer("queue_ptr_slice", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_scan", torch.zeros(1, dtype=torch.long))

        self.image_queue_slice = nn.functional.normalize(self.image_queue_slice, dim=0)
        self.image_queue_scan = nn.functional.normalize(self.image_queue_scan, dim=0)

        # create vision decoder (only compute mim loss for im_q)
        self.vision_decoder = MAEViTDecoder(
            patch_size=self.args.vision_patch_size, embed_dim=self.args.vision_embed_dim,
            depth=self.args.vision_depth, num_heads=self.args.vision_num_heads,
            decoder_embed_dim=self.args.vision_decoder_embed_dim, decoder_depth=self.args.vision_decoder_depth,
            decoder_num_heads=self.args.vision_decoder_num_heads, mlp_ratio=self.args.vision_mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_patches=self.vision_encoder_q.patch_embed.num_patches)

        # create text decoder (for image caption)
        self.text_decoder = ImageCaptioningTransformer(
            backbone=self.args.text_backbone,
            vision_width=self.vision_width,
            vocab_size=self.args.vocab_size,
            num_hidden_layers=self.args.caption_num_hidden_layers,
            output_hidden_states=self.args.caption_output_hidden_states,
            output_attentions=self.args.caption_output_attentions,
            freeze_pretrained=self.args.caption_freeze_pretrained,
            freeze_pretrained_layers=self.args.caption_freeze_pretrained_layers)

        if self.args.tie_text_encoder_decoder:
            tie_encoder_decoder_weights(self.text_encoder_q.bert, self.text_decoder.bert.bert, '', '/attention')
        else:
            tie_encoder_decoder_embeddings(self.text_encoder_q.bert, self.text_decoder.bert.bert, '', '/attention')

        # beam search.
        self.padding_idx = self.args.text_pad_token_id
        self.sos_index = self.args.text_sos_token_id
        self.eos_index = self.args.text_sos_token_id
        self.vocabulary_size = self.args.vocab_size

        self.beam_size = self.args.caption_beam_size
        self.max_steps = self.args.decode_max_length
        self.beam_search = AutoRegressiveBeamSearch(
            self.eos_index, beam_size=self.beam_size, max_steps=self.max_steps
        )

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

    def fn_mim_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.vision_encoder_q.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_valid(self, im, mask_ratio=-1):
        """text: encoded for contrastive learning, caption: decoded for image caption"""
        batch_size, seq_len, c, h, w = im.shape
        im = im.reshape(-1, c, h, w)
        # im_q: (bs*slice, 3, h, w) -> (bs*slice, 1 + pnum, 3, h1, w1) -> mask ->
        # (bs*slice, 1 + pnum * (1-r), 3, h1, w1) -> encoder -> (bs*slice, 1 + pnum * (1-r), d)
        # slice_embed: (bs*slice, 1 + pnum * (1-r), d)
        slice_embed, _, _ = self.vision_encoder_q(im, mask_ratio=0)
        pre_scan_embed = slice_embed[:, 0, :].reshape(batch_size, seq_len, slice_embed.shape[-1])  # (bs, seq, d)
        _, scan_embed = self.vision_fusion_q(pre_scan_embed)

        # masked image modelling
        masked_slice_embed, mask, ids_restore = self.vision_encoder_q(im, mask_ratio=max(mask_ratio, 0))
        masked_slice_pred = self.vision_decoder(masked_slice_embed, ids_restore)
        loss_mim = self.fn_mim_loss(im, masked_slice_pred, mask)

        # image caption
        proj_scan_embed = self.vision_to_text(scan_embed)
        start_predictions = proj_scan_embed.new_full(
            (batch_size,), self.sos_index
        ).long()  # tensor with shape (batch size, ), value=sos_index
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        beam_search_step = functools.partial(
            self.beam_search_step, proj_scan_embed
        )  # beam_search_step: call the function self.beam_search_step and feed the visual_embeddings
        all_top_k_predictions, _ = self.beam_search.search(
            start_predictions, beam_search_step
        )
        gen_caption_ids = all_top_k_predictions[:, 0, :]

        # (batch_size*seq_len, 3, h, w) -> (batch_size, seq_len, 3, h, w)
        im = im.reshape(batch_size, seq_len, c, h, w)
        # (batch_size*seq_len, L, p*p*3) -> (batch_size*seq_len, 3, h, w) -> (batch_size, seq_len, 3, h, w)
        pred_im = self.vision_encoder_q.unpatchify(masked_slice_pred).reshape(batch_size, seq_len, c, h, w)
        pred_im = pred_im.reshape(batch_size, seq_len, c, h, w)
        # (batch_size*seq_len, L) -> (batch_size*seq_len, L, p*p*3) ->
        # (batch_size*seq_len, 3, h, w) -> (batch_size, seq_len, 3, h, w)
        pred_im_mask = mask.unsqueeze(-1).repeat(1, 1, masked_slice_pred.shape[-1])
        pred_im_mask = self.vision_encoder_q.unpatchify(pred_im_mask)
        pred_im_mask = pred_im_mask.reshape(batch_size, seq_len, c, h, w)

        valid_output = {
                  'loss_mim': loss_mim,
                  'pred_caption_ids': gen_caption_ids,
                  'im_gt': im,
                  'im_pred': pred_im,
                  'im_mask': pred_im_mask,
                  }
        return valid_output

    def forward(self, im_q, text_q, text_c=None, im_k=None, text_k=None, mode='train', **kwargs):
        # texts is caption token tensors dict
        assert im_q.shape[2] == 3
        if mode != 'train':
            valid_output = self.forward_valid(im=im_q, mask_ratio=self.args.valid_mask_ratio)
            return valid_output
        with torch.no_grad():
            self.temp_image.clamp_(0.001, 0.5)
            self.temp_multimodal.clamp_(0.001, 0.5)
        device = im_q.device
        batch_size, seq_len, c, h, w = im_q.shape
        im_q = im_q.reshape(-1, c, h, w)
        # im_q: (bs*slice, 3, h, w) -> (bs*slice, 1 + pnum, 3, h1, w1) -> mask ->
        # (bs*slice, 1 + pnum * (1-r), 3, h1, w1) -> encoder -> (bs*slice, 1 + pnum * (1-r), d)
        # slice_embed: (bs*slice, 1 + pnum * (1-r), d)
        slice_embed, mask, ids_restore = self.vision_encoder_q(im_q, mask_ratio=self.mask_ratio_q)
        # slice_feat: (bs*slice, d) -> (bs*slice, d1)
        slice_feat = F.normalize(self.vision_proj_slice_q(slice_embed[:, 0, :]), dim=-1)  # cls token as global feature

        # scan_embed = slice_feat.reshape(batch_size, seq_len, slice_feat.shape[-1])  # (bs, seq, d)
        pre_scan_embed = slice_embed[:, 0, :].reshape(batch_size, seq_len, slice_embed.shape[-1])  # (bs, seq, d)
        # scan_feat = self.avgpool(scan_embed.permute(0, 2, 1)).squeeze(-1)  # (bs, d)
        scan_feat, scan_embed = self.vision_fusion_q(pre_scan_embed)
        scan_attn = torch.ones(scan_embed.size()[:-1], dtype=torch.long).to(device)
        scan_feat = F.normalize(self.vision_proj_scan_q(scan_feat), dim=-1)

        text_output = self.text_encoder_q(text_q, return_dict=True)
        text_embed = text_output.last_hidden_state
        # use mean pooling of the attention tokens as global text feature, (bs, seq_len, d) -> (bs, d)
        text_feat = self.att_mean_pooler(text_embed, text_q['attention_mask'])
        # (bs, d)
        text_feat = F.normalize(self.text_proj_q(text_feat), dim=-1)

        # get vision momentum features
        with torch.no_grad():
            self._momentum_update()
            im_k = im_k.reshape(-1, c, h, w)
            slice_embed_k, _, _ = self.vision_encoder_k(im_k, mask_ratio=self.mask_ratio_k)
            slice_feat_k = F.normalize(self.vision_proj_slice_k(slice_embed_k[:, 0, :]), dim=-1)  # (bs*seq, d)
            slice_feat_all = torch.cat([slice_feat_k.t(), self.image_queue_slice.clone().detach()], dim=1)  # (d, bs*seq+queue)

            pre_scan_embed_k = slice_embed_k[:, 0, :].reshape(batch_size, seq_len, slice_embed_k.shape[-1])  # (bs, seq, d)
            # scan_feat_k = self.avgpool(pre_scan_embed_k.permute(0, 2, 1)).squeeze(-1)  # (bs, d)
            scan_feat_k, scan_embed_k = self.vision_fusion_k(pre_scan_embed_k)
            scan_feat_k = F.normalize(self.vision_proj_scan_k(scan_feat_k), dim=-1)
            scan_feat_all = torch.cat([scan_feat_k.t(), self.image_queue_scan.clone().detach()], dim=1)

        # image-image contrastive: slice
        sim_i2i_slice = slice_feat @ slice_feat_all / self.temp_image  # similarity matrix (bs, queue size)
        labels = torch.arange(slice_feat.shape[0], dtype=torch.long).to(device)
        loss_iic_slice = F.cross_entropy(sim_i2i_slice, labels)

        # image-image contrastive: scan
        sim_i2i_scan = scan_feat @ scan_feat_all / self.temp_image  # similarity matrix (bs, queue size)
        labels = torch.arange(scan_feat.shape[0], dtype=torch.long).to(device)
        loss_iic_scan = F.cross_entropy(sim_i2i_scan, labels)

        # image-text contrastive: scan v.s. text
        gather_scan_feat, gather_text_feat = self._gather_with_grad([scan_feat, text_feat])
        sim_i2t = gather_scan_feat @ gather_text_feat.t() / self.temp_multimodal  # similarity matrix (bs, bs)
        sim_t2i = sim_i2t.t()
        labels = torch.arange(gather_scan_feat.shape[0], dtype=torch.long).to(device)
        loss_itc = (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)) / 2

        self._dequeue_and_enqueue(slice_feat_k, scan_feat_k)

        # masked image modelling
        slice_pred = self.vision_decoder(slice_embed, ids_restore)
        loss_mim = self.fn_mim_loss(im_q, slice_pred, mask)

        # vision-language modelling
        proj_scan_embed = self.vision_to_text(scan_embed)
        output_logits = self.text_decoder(text_c, proj_scan_embed, scan_attn)  # (batch size, seq_len, voc_size)
        # text_q['input_ids']: (batch size, seq_len), long
        t_logits = output_logits[:, :-1, ...].reshape(-1, self.vocabulary_size)
        t_targets = text_c['input_ids'][:, 1:, ...].reshape(-1)
        assert t_logits.shape[0] == t_targets.shape[0]
        loss_lm = F.cross_entropy(t_logits, t_targets, ignore_index=self.padding_idx)
        # gen_caption_logits[:, :-1].reshape(-1, gen_caption_logits.shape[-1]), enc_tokens['input_ids'][:, 1:].reshape(-1)
        gen_caption_ids = torch.argmax(output_logits, dim=-1)  # (batch size, seq_len)
        # attention: argmax eliminates gradients

        output = {'loss_iic_slice': loss_iic_slice,
                  'loss_iic_scan': loss_iic_scan,
                  'loss_mim': loss_mim,
                  'loss_ttc': -1.0 * torch.ones_like(loss_itc, device=loss_itc.device),
                  'loss_itc': loss_itc,
                  'loss_lm': loss_lm,
                  'pred_caption_ids': gen_caption_ids,
                  'sim_iic_slice': sim_i2i_slice,
                  'sim_iic_scan': sim_i2i_scan,
                  'sim_itc_i2t': sim_i2t,
                  'sim_itc_t2i': sim_t2i,
                  }

        return output

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, slice_feat, scan_feat):
        # gather keys before updating queue
        slice_feats = concat_all_gather(slice_feat)  # (bs * seq, d)
        scan_feats = concat_all_gather(scan_feat)  # (bs, d)

        batch_size_slice = slice_feats.shape[0]
        ptr_slice = int(self.queue_ptr_slice)
        assert self.queue_size_slice % batch_size_slice == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue_slice[:, ptr_slice:ptr_slice + batch_size_slice] = slice_feats.T
        ptr_slice = (ptr_slice + batch_size_slice) % self.queue_size_slice  # move pointer

        self.queue_ptr_slice[0] = ptr_slice

        batch_size_scan = scan_feats.shape[0]
        ptr_scan = int(self.queue_ptr_scan)
        assert self.queue_size_scan % batch_size_scan == 0  # for simplicity
        assert self.queue_size_scan % batch_size_scan == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue_scan[:, ptr_scan:ptr_scan + batch_size_scan] = scan_feats.T
        ptr_scan = (ptr_scan + batch_size_scan) % self.queue_size_scan  # move pointer

        self.queue_ptr_scan[0] = ptr_scan

    def _gather_with_grad(self, feats):
        """feats: list of tensor which needed to be gather respectively"""
        if not isinstance(feats, list):
            feats = [feats]
        gather_feats = []
        for feat in feats:
            gather_feat = link.AllGather.apply(feat)
            gather_feat = gather_feat.view(-1, *(feat.shape[1:]))
            gather_feats.append(gather_feat)
        return gather_feats

    def beam_search_step(self, visual_features: torch.Tensor, partial_captions: torch.Tensor) -> torch.Tensor:
        r"""
        Given visual features and a batch of (assumed) partial captions, predict
        the distribution over vocabulary tokens for next time-step. This method
        is used by :class:`~refers.utils.beam_search.AutoRegressiveBeamSearch`.

        Parameters
        ----------
        projected_visual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., textual_feature_size)``
            with visual features already projected to ``textual_feature_size``.
        partial_captions: torch.Tensor
            A tensor of shape ``(batch_size * beam_size, timesteps)``
            containing tokens predicted so far -- one for each beam. We need all
            prior predictions because our model is auto-regressive.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size * beam_size, vocab_size)`` -- output
            distribution over tokens for next time-step.
        """

        # Expand and repeat image features while doing beam search.
        batch_size, patch_num, feature_size = visual_features.size()
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, patch_num, feature_size
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        # caption_lengths = torch.ones_like(partial_captions)

        # if len(caption_lengths.size()) == 2:
        #     caption_lengths = caption_lengths.sum(1)  # (batch size,)
        # else:
        #     # Add a time-step. shape: (batch_size, 1)
        #     partial_captions = partial_captions.unsqueeze(1)
        if len(partial_captions.size()) != 2:
            partial_captions = partial_captions.unsqueeze(1)

        padding_mask = torch.ones((partial_captions.shape[0], partial_captions.shape[1])).to(
            partial_captions.device)

        # padding_mask = torch.cat([torch.ones((partial_captions.shape[0], partial_captions.shape[1])), torch.zeros(
        #     (partial_captions.shape[0], self.config.tokenizer.max_length - partial_captions.shape[1]))], dim=1).to(
        #     partial_captions.device)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        enc_captions = {'input_ids': partial_captions, 'attention_mask': padding_mask}

        # print('(train) text input size: ', enc_captions['input_ids'].shape)
        # print('(train) text padding mask size: ', enc_captions['attention_mask'].shape)

        output_logits = self.text_decoder(enc_captions, visual_features,)  # (bs, max_len, voc_size)
        # Keep features for last time-step only, we only care about those.
        output_logits = output_logits[:, -1, :]

        # Return logprobs as required by `AutoRegressiveBeamSearch`.
        # shape: (batch_size * beam_size, vocab_size)
        next_logprobs = F.log_softmax(output_logits, dim=1)

        # Set logprobs of last predicted tokens as high negative value to avoid
        # repetition in caption.
        for index in range(batch_size * beam_size):
            next_logprobs[index, partial_captions[index, -1]] = -10000

        return next_logprobs


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)


def tie_encoder_decoder_embeddings(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name and 'embeddings' in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)

StateType = Dict[str, torch.Tensor]
StepFunctionType = Callable[..., torch.Tensor]

class AutoRegressiveBeamSearch(object):
    r"""
    Implements the beam search algorithm for decoding the most likely captions.
    This only works for auto-regressive models (Transformer-like) and not
    recurrent models (LSTM-like).

    Parameters
    ----------
    end_index: int
        The index of the end token (``[EOS]``) in vocabulary.
    max_steps: int, optional (default = 50)
        The maximum number of decoding steps.
    beam_size: int, optional (default = 5)
        The width of the beam used.
    per_node_beam_size: int, optional (default = 2)
        The maximum number of candidates to consider per node, at each step in
        the search. Setting this parameter to a number smaller than `beam_size`
        may give better results, as it can introduce more diversity into the
        search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(
            self,
            end_index: int,
            max_steps: int = 50,
            beam_size: int = 5,
            per_node_beam_size: int = 2,
    ) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        # self.per_node_beam_size = per_node_beam_size or beam_size
        self.per_node_beam_size = beam_size
        # print('per node beam size = ', self.per_node_beam_size)

    def search(
            self, start_predictions: torch.Tensor, step: StepFunctionType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given a starting state and a step function, apply beam search to find
        the most likely target captions.

        .. note::

            If your step function returns ``-inf`` for some log probs
            (like if you're using a masked log-softmax) then some of the "best"
            captions returned may have ``-inf`` log probability. Specifically
            this happens when the beam size is smaller than the number of actions
            with finite log probability (non-zero probability) returned by the
            step function. Therefore if you're using a mask you may want to
            check the results from ``search`` and potentially discard captions
            with non-finite log probability.

        Parameters
        ----------
        start_predictions : torch.Tensor
            Tensor containing the initial predictions, shape ``(batch_size, )``.
            Usually the initial predictions are just the index of the start
            token (``[SOS]``) in the vocabulary.
        step : Callable[..., torch.Tensor]
            A function that is responsible for computing the next most likely
            tokens, given the past predictions. Predictions from all previous
            time-steps are required, not just the last time-step, because our
            model is auto-regressive instead of recurrent.  The function should
            The function is expected to return a tensor of shape
            ``(group_size, target_vocab_size)`` containing
            the log probs of the tokens for the next step.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probs)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probs``
            has shape ``(batch_size, beam_size)``.
        """
        batch_size = start_predictions.size()[0]

        # List of `(batch_size, beam_size)` tensors. One for each time step.
        # Does not include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None
        # for the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probs = step(start_predictions)

        num_classes = start_class_log_probs.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ValueError(
                f"Target vocab size ({num_classes:d}) too small "
                f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                f"Please decrease beam_size or per_node_beam_size."
            )

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probs, start_predicted_classes = start_class_log_probs.topk(
            self.beam_size
        )
        if (
                self.beam_size == 1
                and (start_predicted_classes == self._end_index).all()
        ):
            warnings.warn(
                "Empty captions predicted. You may want to increase beam "
                "size or ensure your step function is working properly.",
                RuntimeWarning,
            )
            return start_predicted_classes.unsqueeze(-1), start_top_log_probs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probs = start_top_log_probs

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes.
            predictions_so_far = torch.stack(predictions).permute(1, 2, 0).view(
                batch_size * self.beam_size, -1
            )
            # shape: (batch_size * beam_size, num_classes)
            class_log_probs = step(predictions_so_far)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )
            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probs = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probs,
            )
            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probs, predicted_classes = cleaned_log_probs.topk(
                self.per_node_beam_size
            )
            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probs = (
                last_log_probs.unsqueeze(2)
                    .expand(batch_size, self.beam_size, self.per_node_beam_size)
                    .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probs = top_log_probs + expanded_last_log_probs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices
            )
            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probs = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size`
            # dimension where the indices with a common ancestor are grouped
            # together. Hence dividing by `per_node_beam_size` gives the
            # ancestor. (Note that this is integer division as the tensor is a
            # LongTensor.)
            # shape: (batch_size, beam_size)
            # print(restricted_beam_indices)

            # backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointer = torch.div(restricted_beam_indices, self.per_node_beam_size, rounding_mode='trunc')

            backpointers.append(backpointer)

        if not torch.isfinite(last_log_probs).all():
            warnings.warn(
                "Infinite log probs encountered. Some final captions may not "
                "make sense. This can happen when the beam size is larger than"
                " the number of valid (non-zero probability) transitions that "
                "the step function produces.",
                RuntimeWarning,
            )

        # Reconstruct the captions.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            # print(cur_backpointers.shape)
            cur_preds = (
                predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            )
            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probs
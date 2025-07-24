# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 20:28

import os
import torch
from torch import nn
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
import warnings
import functools
from functools import partial
import torch.nn.functional as F
import transformers
from timm.models.layers import trunc_normal_
transformers.logging.set_verbosity_error()
from .modules.vit import VisionTransformer
from .modules.pos_embed import interpolate_pos_embed
from .modules.fusion_modules import FusionTransformer
from .modules.bert import ImageCaptioningTransformer


class CaptionModel(nn.Module):
    def __init__(self, args, stage='train',**kwargs):
        super(CaptionModel, self).__init__()
        # define parameters
        self.args = args
        self.args = args
        self.stage = stage
        self.vision_encoder = VisionTransformer(
            patch_size=self.args.vision_patch_size, embed_dim=self.args.vision_embed_dim,
            depth=self.args.vision_depth, num_heads=self.args.vision_num_heads, num_classes=-1,
            mlp_ratio=self.args.vision_mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pretrained=self.args.vision_pretrained, checkpoint=self.args.vision_pretrained_weight,
            freeze_pretrained_layers=self.args.vision_freeze_pretrained_layers,
        )
        self.vision_width = self.args.vision_embed_dim
        self.vision_fusion = FusionTransformer(embed_dim=self.vision_width)

        if self.stage == 'train' and self.args.vision_load_dir != '':
            self.load_vision_checkpoint()

        # create text decoder (for image caption)
        self.text_decoder = ImageCaptioningTransformer(
            backbone=self.args.caption_backbone,
            vision_width=self.vision_width,
            vocab_size=self.args.vocab_size,
            num_hidden_layers=self.args.caption_num_hidden_layers,
            output_hidden_states=self.args.caption_output_hidden_states,
            output_attentions=self.args.caption_output_attentions,
            pretrained=self.args.caption_pretrained,
            freeze_pretrained=self.args.caption_freeze_pretrained,
            freeze_pretrained_layers=self.args.caption_freeze_pretrained_layers)
        self.vision_to_text = nn.Linear(self.vision_width, self.text_decoder.config.hidden_size)

        if self.stage == 'train' and self.args.caption_load_dir != '' and self.args.model != "v7":
            self.load_text_decoder_checkpoint()

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

        self.criterion = nn.CrossEntropyLoss()

        if self.args.linear_probe:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.vision_fusion.parameters():
                param.requires_grad = False
            for param in self.text_decoder.parameters():
                param.requires_grad = False
            for param in self.vision_to_text.parameters():
                param.requires_grad = False

    def load_vision_checkpoint(self, load_path=""):
        if load_path == "":
            load_path = os.path.join(self.args.vision_load_dir,
                                     f'checkpoints/checkpoint_{self.args.vision_load_epoch}.pth')
        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load vision encoder from pre-trained checkpoint: %s" % load_path)
        checkpoint_model = checkpoint['model']
        ### load vision model
        vision_pretrained_weights = OrderedDict()
        vision_fusion_pretrained_weights = OrderedDict()
        for k, v in checkpoint_model.items():
            if k.startswith('module.vision_encoder_q.'):
                vision_pretrained_weights[k[len('module.vision_encoder_q.'):]] = v
            if k.startswith('module.vision_fusion_q.'):
                vision_fusion_pretrained_weights[k[len('module.vision_fusion_q.'):]] = v
        vision_state_dict = self.vision_encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in vision_pretrained_weights and vision_pretrained_weights[k].shape != vision_state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del vision_pretrained_weights[k]

        # interpolate position embedding
        interpolate_pos_embed(self.vision_encoder, vision_pretrained_weights)

        # load pre-trained model
        msg = self.vision_encoder.load_state_dict(vision_pretrained_weights, strict=False)
        print(msg)
        if 'head.weight' in vision_state_dict or 'head.bias' in vision_state_dict:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            # manually initialize fc layer
            trunc_normal_(self.vision_encoder.head.weight, std=2e-5)

        self.vision_fusion.load_state_dict(vision_fusion_pretrained_weights, strict=True)
        print("Load vision fusion encoder from pre-trained checkpoint: %s" % load_path)

    def load_text_decoder_checkpoint(self, load_path=""):
        if load_path == "":
            load_path = os.path.join(self.args.caption_load_dir,
                                     f'checkpoints/checkpoint_{self.args.caption_load_epoch}.pth')
        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load text decoder from pre-trained checkpoint: %s" % load_path)
        checkpoint_model = checkpoint['model']
        caption_pretrained_weights = OrderedDict()
        for k, v in checkpoint_model.items():
            if k.startswith('module.text_decoder.'):
                caption_pretrained_weights[k[len('module.text_decoder.'):]] = v

        # load pre-trained model
        msg = self.text_decoder.load_state_dict(caption_pretrained_weights, strict=False)
        print(msg)

        if self.args.vision_load_dir == self.args.caption_load_dir:
            print("Load vision_to_text head from pre-trained checkpoint: %s" % load_path)
            vision_to_text_pretrained_weights = OrderedDict()
            for k, v in checkpoint_model.items():
                if k.startswith('module.vision_to_text.'):
                    vision_to_text_pretrained_weights[k[len('module.vision_to_text.'):]] = v
            # load pre-trained model
            msg = self.vision_to_text.load_state_dict(vision_to_text_pretrained_weights, strict=False)
            print(msg)

    def forward(self, im, text, mode='train', **kwargs):
        # texts is caption token tensors dict
        assert im.shape[2] == 3
        device = im.device
        batch_size, seq_len, c, h, w = im.shape
        im = im.reshape(-1, c, h, w)
        # im: (bs*slice, 3, h, w) -> (bs*slice, 1 + pnum, 3, h1, w1) -> mask ->
        # (bs*slice, 1 + pnum * (1-r), 3, h1, w1) -> encoder -> (bs*slice, 1 + pnum * (1-r), d)
        # slice_embed: (bs*slice, d)
        slice_embed = self.vision_encoder(im)
        scan_embed = slice_embed.reshape(batch_size, seq_len, slice_embed.shape[-1])  # (bs, seq, d)
        if self.args.model not in ["v1", "v8"]:
            _, scan_embed = self.vision_fusion(scan_embed)
        scan_attn = torch.ones(scan_embed.size()[:-1], dtype=torch.long).to(device)

        # vision-language modelling
        proj_scan_embed = self.vision_to_text(scan_embed)
        if mode == 'train':
            output_logits = self.text_decoder(text, proj_scan_embed, scan_attn)  # (batch size, seq_len, voc_size)
            t_logits = output_logits[:, :-1, ...].reshape(-1, self.vocabulary_size)
            t_targets = text['input_ids'][:, 1:, ...].reshape(-1)
            assert t_logits.shape[0] == t_targets.shape[0]
            loss = F.cross_entropy(t_logits, t_targets, ignore_index=self.padding_idx)
            # gen_caption_logits[:, :-1].reshape(-1, gen_caption_logits.shape[-1]), enc_tokens['input_ids'][:, 1:].reshape(-1)
            gen_caption_ids = torch.argmax(output_logits, dim=-1)  # (batch size, seq_len)
            # attention: argmax eliminates gradients
        elif mode == 'vis':
            output_logits, output_crossattns = self.text_decoder(text, proj_scan_embed, scan_attn, output_attn=True)  # (batch size, seq_len, voc_size)
            return output_logits[:, :-1, ...].reshape(-1, self.vocabulary_size), output_crossattns
        else:
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
            loss = -1

        output = {
                  'loss': loss,
                  'pred_caption_ids': gen_caption_ids,
                  }

        return output

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







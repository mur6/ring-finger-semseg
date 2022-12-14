import math
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as transforms
from transformers import SegformerDecodeHead, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerMLP, SegformerModel


class MySegformerDecodeHead(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.classifier = nn.Linear(768 * 128 * 128, 4)
        # self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        count = len(all_hidden_states)
        print("after all_hidden_states: length: ", count)
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        print("after linear_fuse: nn.Conv2d: ", hidden_states.shape)
        hidden_states = self.batch_norm(hidden_states)
        print("after batch_norm: nn.BatchNorm2d: ", hidden_states.shape)
        hidden_states = self.activation(hidden_states)
        print("after activation: nn.ReLU: ", hidden_states.shape)
        hidden_states = hidden_states.contiguous().view(batch_size, -1)
        print("after contiguous.view: ", hidden_states.shape)
        hidden_states = self.dropout(hidden_states)
        print("after dropout: nn.Dropout: ", hidden_states.shape)
        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)
        print("after classifier: nn.Linear: ", logits.shape)
        return logits


class MySegformerForSemanticSegmentation(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = MySegformerDecodeHead(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)

        return logits


def get_model():
    id2label = {0: "unlabeled", 1: "points"}
    label2id = {v: k for k, v in id2label.items()}
    model = MySegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        ignore_mismatched_sizes=True,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        reshape_last_stage=True,
    )
    return model


if __name__ == "__main__":
    model = get_model()
    out = model(torch.rand(1, 3, 512, 512))
    print(out.shape)
    # print(model)

import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage import measure
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as transforms
from torchvision.transforms import Resize, ToTensor
from transformers import SegformerDecodeHead, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerMLP, SegformerModel

albumentations_transform = A.Compose(
    [
        A.Resize(224, 224),
    ]
)

from dataset2 import RingFingerDataset

# base_data_dir = Path("datasets")
base_data_dir = Path("../blender-for-finger-segmentation/data2/")

# from dataset import ImageSegmentationDataset
# train_dataset = ImageSegmentationDataset(
#     root_dir=base_data_dir / "training", feature_extractor=feature_extractor_inference, transforms=None
# )
# valid_dataset = ImageSegmentationDataset(
#     root_dir=base_data_dir / "validation", feature_extractor=feature_extractor_inference, transforms=None, train=False
# )

# from dataset import ImageSegmentationDataset


feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)


train_dataset = RingFingerDataset(
    base_data_dir / "training",
    "data/datasets/contour_checked_numbers_training.json",
    feature_extractor=feature_extractor_inference,
    transform=None,
)
valid_dataset = RingFingerDataset(
    base_data_dir / "validation",
    "data/datasets/contour_checked_numbers_validation.json",
    feature_extractor=feature_extractor_inference,
    transform=None,
)


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

        self.classifier = nn.Linear(12582912, 4)
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

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.contiguous().view(batch_size, -1)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b2")
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2")

id2label = {0: "unlabeled", 1: "hand", 2: "mat"}
label2id = {v: k for k, v in id2label.items()}


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


from torch import nn
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

model = MySegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2",
    ignore_mismatched_sizes=True,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    reshape_last_stage=True,
)

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AdamW

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=0.00006)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model Initialized!")

# pbar = tqdm(train_dataloader)

for epoch in range(1, 1 + 1):
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    model.train()
    train_loss = 0.0
    for idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        points = batch["points"].to(device)
        print(pixel_values.shape, points.shape)
        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values)
        # print()
        # print(outputs.dtype)

        # evaluate
        points = ((points - (224 / 2.0)) / 112.0).float()

        # pred_labels = predicted[mask].detach().cpu().numpy()
        # true_labels = labels[mask].detach().cpu().numpy()
        # accuracy = accuracy_score(pred_labels, true_labels)
        # accuracies.append(accuracy)
        # pbar.set_postfix(
        #     {"Batch": idx, "Pixel-wise accuracy": sum(accuracies) / len(accuracies), "Loss": sum(losses) / len(losses)}
        # )

        # backward + optimize
        loss = criterion(outputs, points)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # pixel_values = batch[0].to(device)
        # # maskes = batch[1].to(device)
        # points = batch[2].to(device)

        # # zero the parameter gradients
        # optimizer.zero_grad()

        # # forward
        # outputs = model(pixel_values=pixel_values)

        # # evaluate
        # upsampled_logits = nn.functional.interpolate(
        #     outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        # )
        # predicted = upsampled_logits.argmax(dim=1)

        # pred_labels = predicted[mask].detach().cpu().numpy()
        # true_labels = labels[mask].detach().cpu().numpy()
        # accuracy = accuracy_score(pred_labels, true_labels)
        # loss = outputs.loss
        # accuracies.append(accuracy)
        # losses.append(loss.item())
        # pbar.set_postfix(
        #     {"Batch": idx, "Pixel-wise accuracy": sum(accuracies) / len(accuracies), "Loss": sum(losses) / len(losses)}
        # )

        # # backward + optimize
        # loss.backward()
        # optimizer.step()
    else:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                masks = batch[1]
                masks[masks == 2] = -1
                masks = masks.to(device)
                points = batch[2].to(device)
                points = (points - (224 / 2.0)) / 112.0
                outputs = model(masks=masks)
                loss = criterion(outputs, points)
                val_loss += loss.item()
        # with torch.no_grad():
        #     for idx, batch in enumerate(valid_dataloader):
        #         pixel_values = batch["pixel_values"].to(device)
        #         labels = batch["labels"].to(device)

        #         outputs = model(pixel_values=pixel_values, labels=labels)
        #         upsampled_logits = nn.functional.interpolate(
        #             outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        #         )
        #         predicted = upsampled_logits.argmax(dim=1)

        #         mask = labels != 0  # we don't include the background class in the accuracy calculation
        #         pred_labels = predicted[mask].detach().cpu().numpy()
        #         true_labels = labels[mask].detach().cpu().numpy()
        #         accuracy = accuracy_score(pred_labels, true_labels)
        #         val_loss = outputs.loss
        #         val_accuracies.append(accuracy)
        #         val_losses.append(val_loss.item())
    # writer.add_scalar('Loss/train', sum(losses)/len(losses), epoch)
    # writer.add_scalar('Loss/val', sum(val_losses)/len(val_losses), epoch)
    # writer.add_scalar('Accuracy/train', sum(accuracies)/len(accuracies), epoch)
    # writer.add_scalar('Accuracy/val', sum(val_accuracies)/len(val_accuracies), epoch)
    # print(
    #     f"Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}\
    #      Train Loss: {sum(losses)/len(losses)}\
    #      Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}\
    #      Val Loss: {sum(val_losses)/len(val_losses)}"
    # )
    train_count = len(train_dataloader)
    val_count = len(valid_dataloader)
    s1 = f"Training: Mean Squared Error: {train_loss/train_count}"
    s2 = f"Validation: Mean Squared Error: {val_loss/val_count}"
    print(s1 + " " + s2)

from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np

base_data_dir = Path("../blender-for-finger-segmentation/")
# print(base_dir)

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
# print(feature_extractor)


from dataset import ImageSegmentationDataset

train_dataset = ImageSegmentationDataset(
    root_dir=base_data_dir / "training", feature_extractor=feature_extractor, transforms=None
)
valid_dataset = ImageSegmentationDataset(
    root_dir=base_data_dir / "validation", feature_extractor=feature_extractor, transforms=None, train=False
)

encoded_inputs = train_dataset[0]
print(encoded_inputs["pixel_values"].shape)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4)

# classes = pd.read_csv('./drone_dataset/class_dict_seg.csv')['name']
id2label = {0: "unlabeled", 1: "hand", 2: "mat"}
# id2label = classes.to_dict()
label2id = {v: k for k, v in id2label.items()}

print(label2id)

# import matplotlib.pyplot as plt

# mask = encoded_inputs["labels"].numpy()
# plt.imshow(mask)
# plt.show()

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1",
    ignore_mismatched_sizes=True,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    reshape_last_stage=True,
)
import torch
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=0.00006)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model Initialized!")

for epoch in range(1, 15 + 1):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    model.train()
    for idx, batch in enumerate(pbar):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(pixel_values=pixel_values, labels=labels)

        # evaluate
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)

        mask = labels != 0  # we don't include the background class in the accuracy calculation
        # print(mask)
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        loss = outputs.loss
        accuracies.append(accuracy)
        losses.append(loss.item())
        pbar.set_postfix(
            {"Batch": idx, "Pixel-wise accuracy": sum(accuracies) / len(accuracies), "Loss": sum(losses) / len(losses)}
        )

        # backward + optimize
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                upsampled_logits = nn.functional.interpolate(
                    outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)

                mask = labels != 0  # we don't include the background class in the accuracy calculation
                pred_labels = predicted[mask].detach().cpu().numpy()
                true_labels = labels[mask].detach().cpu().numpy()
                accuracy = accuracy_score(pred_labels, true_labels)
                val_loss = outputs.loss
                val_accuracies.append(accuracy)
                val_losses.append(val_loss.item())
    # writer.add_scalar('Loss/train', sum(losses)/len(losses), epoch)
    # writer.add_scalar('Loss/val', sum(val_losses)/len(val_losses), epoch)
    # writer.add_scalar('Accuracy/train', sum(accuracies)/len(accuracies), epoch)
    # writer.add_scalar('Accuracy/val', sum(val_accuracies)/len(val_accuracies), epoch)
    print(
        f"Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}\
         Train Loss: {sum(losses)/len(losses)}\
         Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}\
         Val Loss: {sum(val_losses)/len(val_losses)}"
    )
# writer.flush()

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

model_dir = "models/segformer_b2/"
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval()
# inputs = feature_extractor_inference(images=images, return_tensors="pt")
# outputs = model(**inputs)
# # shape (batch_size, num_labels, height, width)
# logits = outputs.logits
# print(f"outputs: {type(outputs)}")
# print(f"outputs.logis: {logits.shape}")

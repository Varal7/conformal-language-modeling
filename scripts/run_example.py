import os
import numpy as np
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from clm.components import compute_values_for_C_inner, compute_values_for_C_inner_basic

base = "data/cxr/val"

rouge_label = np.load(os.path.join(base, "components", "rouge_label.npy"))
rouge_with_refs = np.load(os.path.join(base, "components", "rouge_with_refs.npy"))
image_preds = np.load(os.path.join(base, "components", "image_preds.npy"))
mask = np.load(os.path.join(base, "components", "mask.npy"))
oracle_size = np.load(os.path.join(base, "components", "oracle_size.npy"))

# New version

results = compute_values_for_C_inner_basic(image_preds, rouge_label, mask, oracle_size, rouge_threshold=0.4)

fig, ax = plt.subplots(1, 1)
ax.plot(results['L_avg'], results['fake_recall'], marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*| / |C^*|]$")
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 2)
ax.set_title(f"C_inner (fake recall), with rougeL_threshold = 0.4")
plt.show()

# Old version

results = compute_values_for_C_inner(image_preds, rouge_with_refs, rouge_threshold=0.4)

fig, ax = plt.subplots(1, 1)
ax.plot(results['L_avg'], results['fake_recall'], marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*| / |C^*|]$")
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 2)
ax.set_title(f"C_inner (fake recall), with rougeL_threshold = 0.4")
plt.show()

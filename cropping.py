# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import pandas as pd
import cv2
import openslide

import matplotlib.pyplot as plt
import seaborn as sns

from utils import crop_tissues

# %%
# Open slide
slide = openslide.open_slide("/workspaces/ATPase/data/DLM - ATPase Alexandra/Raw/Slide 1.ndpi")

print(f"slide.level_count: {slide.level_count}")
print(f"slide.level_dimensions: {slide.level_dimensions}")
print(f"slide.level_downsamples: {slide.level_downsamples}")

# Get thumbnail of level 3
slide.get_thumbnail(size=slide.level_dimensions[-3])

# %%
path = '/workspaces/ATPase/data/DLM - ATPase Alexandra/Raw/'
results_dir = '/workspaces/ATPase/data/DLM - ATPase Alexandra/Processed/'

# List the slides
slides = [slide for slide in os.listdir(path)]

for slide_name in slides:
    slide_path = path + slide_name
    # check if the slide is already processed
    if os.path.exists(results_dir + slide_name.replace('.ndpi', '')):
        print(f"Slide {slide_name} already processed")
        continue
    else:
        print(f"Processing {slide_name}")
        crop_tissues(slide_path,
            level=5,
            min_area_ratio=0.03,
            min_hole_ratio=0.03,
            hsv_lower=(0, 0, 0),
            hsv_upper=(150, 150, 150),
            show_steps=True,
            show_results=True,
            results_dir=results_dir
    )

# %%
cropped_tissues = crop_tissues(
    slide_path="/workspaces/ATPase/data/DLM - ATPase Alexandra/Raw/Slide 20.ndpi",
    level=5,
    min_area_ratio=0.05,
    min_hole_ratio=0.02,
    hsv_lower=(0, 0, 0),
    hsv_upper=(90, 90, 90),
    show_steps=True,
    show_results=True
)

# %%
slide_tissue = openslide.open_slide("/workspaces/ATPase/Slide 1/5568_29328.ome.tiff")
# %%
print(f"slide_tissue.level_count: {slide_tissue.level_count}")
print(f"slide_tissue.level_dimensions: {slide_tissue.level_dimensions}")
print(f"slide_tissue.level_downsamples: {slide_tissue.level_downsamples}")

# %%

path = "/workspaces/ATPase/Slide 1/5568_29328.ome.tiff"

with tiff.TiffFile(path) as tf:
    levels = tf.series[0].levels   # SubIFD pyramid levels
    print("num_levels:", len(levels))
    print("shapes:", [lvl.shape for lvl in levels])

    # load a lower-res level (e.g., level 3)
    # img_lvl3 = levels[3].asarray()  # Y, X, C
# %%
levels
# %%
import tifffile as tiff

path = "/workspaces/ATPase/Slide 1/5568_29328.ome.tiff"

slide = tiff.TiffFile(path)
slide_levels = slide.series[0].levels
shapes = [lvl.shape for lvl in slide_levels]

print(shapes)
# %%
img = slide_levels[-2].asarray()
plt.imshow(img)
plt.show()
# %%
# Grayscale image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img_gray.shape)

# Plot the image and a histogram of the image
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title('Grayscale Image')
ax[0].axis('off')
fig.colorbar(ax[0].imshow(img_gray, cmap='gray'))
ax[1].hist(img_gray.flatten(), bins=256)
ax[1].set_title('Histogram of Grayscale Image')
ax[1].grid(True)
plt.show()
# %%
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(img_hsv.shape)
# Plot the image and a histogram of the image
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(img_hsv, cmap='hsv')
ax[0].set_title('HSV Image')
ax[0].axis('off')
fig.colorbar(ax[0].imshow(img_hsv, cmap='hsv'))
ax[1].hist(img_hsv.flatten(), bins=256)
ax[1].set_title('Histogram of HSV Image')
ax[1].grid(True)
plt.show()
# %%
img_transformed = cv2.cvtColor(img, cv2.COLOR_BGR2L)
print(img_transformed.shape)
# Plot the image and a histogram of the image
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(img_transformed, cmap='hsv')
ax[0].set_title('HSV Image')
ax[0].axis('off')
fig.colorbar(ax[0].imshow(img_transformed, cmap='hsv'))
ax[1].hist(img_transformed.flatten(), bins=256)
ax[1].set_title('Histogram of HSV Image')
ax[1].grid(True)
plt.show()
# %%
cv2.COLOR_BGR2HLS_FULL
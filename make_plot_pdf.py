import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Name of the model folder")
args = parser.parse_args()

model_name = args.model_name

# A4 paper size in inches (portrait)
fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))

base_path = os.path.join("results", model_name, "plots")

files = [
    (os.path.join(base_path, "dice_train.png"), "Training Dice"),
    (os.path.join(base_path, "dice_val.png"), "Validation Dice"),
    (os.path.join(base_path, "loss_train.png"), "Training Loss"),
    (os.path.join(base_path, "loss_val.png"), "Validation Loss"),
]

for ax, (fname, title) in zip(axes.flatten(), files):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

plt.tight_layout()
out_path = os.path.join("results", model_name, "plot.pdf")
plt.savefig(out_path)
print(f"Saved combined PDF at {out_path}")

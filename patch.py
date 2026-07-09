import sys
import os
import re
import numpy as np
from PIL import Image

root = sys.argv[1]
directory = os.fsencode(root)
max_hori = 8
max_verti = 4

images = []
files = sorted(os.listdir(directory), key=lambda f: int(re.search(r'\d+', os.fsdecode(f)).group()))

for file in files:
    img_name = os.fsdecode(file)
    fileDir = os.path.join(root, img_name)
    images.append(Image.open(fileDir))

rows = []
for i in range(max_verti):
    row_imgs = []
    for j in range(max_hori):
        index = i * max_hori + j
        row_imgs.append(np.array(images[index]))
    rows.append(np.hstack(row_imgs))   # hstack a plain list of arrays, no extra dimension needed

output_path = os.path.join(root, "hologram32.png")
final_img = np.vstack(rows)   # vstack the row arrays directly
PIL_image = Image.fromarray(final_img.astype(np.uint8))
PIL_image.save(output_path)
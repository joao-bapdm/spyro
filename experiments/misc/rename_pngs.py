"""This script sweeps through output directories and renames png's accordingly"""

import os

# get relevant directories
dirs = [f for f in os.scandir() if f.is_dir()]
outdirs = [dir for dir in dirs if "output" in dir.name]

for dir in outdirs:
    # register prefix
    prefix = dir.name[dir.name.find("_") + 1:]
    # get image file
    images = [f for f in os.listdir(dir) if f.endswith(".png")]
    image = os.path.join(dir, images.pop()) if images else None
    # rename image
    new_name = os.path.join(dir, prefix + ".png")
    print(f"\nrenaming image in {dir.path}")
    print(f"the old image is {image}")
    print(f"the new image is {new_name}")
    if image:
        os.rename(image, new_name)


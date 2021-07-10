"""Load image file by using Pillow and matplotlib"""

import h5py
import numpy as np
from PIL import Image

def load_salt(image="salt.png"):
    """Load salt image and return numpy array"""
    
    # Load image and convert to grayscale
    im = Image.open(image).convert(mode="L")
    # Trnsform to array
    arr = np.asarray(im)
    # Reduce to two materials
    hist, bin_edges = np.histogram(arr, bins=2)
    threshold = bin_edges[1]
    # Locate salt
    salt = arr > threshold
    vp_salt = np.where(salt, 1, 0)

    return vp_salt

if __name__ == '__main__':

    # hdf5 destination file
    h5file = "kadu_salt.hdf5"
    # Load salt
    vp_salt = load_salt()
    # Define model geometry
    nz, nx = vp_salt.shape
    depth, width = 3, 10
    # Create grid
    zp = np.linspace(0, depth, nz)
    xp = np.linspace(0, width, nx)
    X, Z = np.meshgrid(xp, zp)
    # Get coordinates
    # coords = np.array([X.flatten(), Z.flatten()]).transpose()
    coords = np.array([Z.flatten(), X.flatten()]).transpose()

    print("Writing velocity model: " + h5file, flush=True)
    with h5py.File(h5file, "w") as f:
        f.create_dataset("velocity_model", data=vp_salt.flatten(), dtype="f")
        f.create_dataset("coordinates", data=coords, dtype="f")
        f.attrs["geometric dimension"] = coords.shape[1]


import numpy as np
import imageio

import sys
from os.path import isfile
import struct

def main(argv):
    if len(argv) != 2:
        print("Usage: cat_decoder.py <input_file> <output_file>")
        sys.exit(1)
    inCat, outCat = argv[0], argv[1]
    if not isfile(inCat):
        print("{} is not a valid file".format(inCat))
        sys.exit(2)
    print("Going to decode {} to output file {}".format(inCat, outCat))

    print("Reading file {}...".format(inCat), end='', flush=True)
    f = open(inCat, "rb")
    print("done!")

    print("Detecting...", end='', flush=True)
    imHeight = struct.unpack('i', f.read(4))[0]
    imWidth = struct.unpack('i', f.read(4))[0]
    N = struct.unpack('B', f.read(1))[0]
    dCat = np.zeros([imHeight, imWidth], dtype=np.uint8)
    nPoints = (imHeight * imWidth) // N
    print("Shape {}x{}...".format(imWidth, imHeight), end='', flush=True)

    centroids = np.zeros([N ** 2, N], dtype=np.uint8)
    print("{} Centroids!".format(centroids.shape[0]))

    print("Extracting Centroids...", end='', flush=True)
    for i in range(N ** 2):
        for j in range(N):
            centroids[i][j] = struct.unpack('B', f.read(1))[0]
    print("done!")

    print("Unpacking bit bundle", end='', flush=True)
    residualrows = imHeight % 2
    resres = imWidth * residualrows // N
    residual = (nPoints + resres) % 2
    bitUnpacker = np.zeros([(nPoints + resres) // 2 + residual], dtype=np.uint8)
    dCat1D = np.zeros([nPoints + resres], dtype=np.uint8)
    print(".", end='', flush=True)
    """ The ideal mechanism is technically like this
        |  A  --  B  |
        | 0000  0000 |
        <--- byte --->
    """
    for i in range((nPoints + resres) // 2 + residual):
        bitUnpacker[i] = struct.unpack('B', f.read(1))[0]
    f.close()

    print(".", end='', flush=True)

    idx = 0
    for i in range(0, (nPoints + resres) - residual, 2):
        dCat1D[i + 0] = (bitUnpacker[idx + 0] & 0b11110000) >> 4
        dCat1D[i + 1] = (bitUnpacker[idx + 0] & 0b00001111)
        idx += 1

    add = 0
    if residual > 0:
        dCat1D[(nPoints + resres) - residual] = (bitUnpacker[idx] & 0b11110000) >> 4
        add += 1
    idx += add

    print("done!")
    print("Writing result on file {}...".format(outCat), end='', flush=True)
    if not isfile(outCat):
        open(outCat, "x").close()
    print("done!")

    # Direct map to the representative value
    # Replacement via index (3 bits ID)
    idx = 0
    for i in range(0, imHeight - residualrows - 1, 2):
        for j in range(0, imWidth, 2):
            dCat[i : i + 2, j : j + 2] = np.array(centroids[dCat1D[idx]][0 : N]).reshape([2, 2])
            idx += 1

    if residualrows > 0:
        for j in range(0, imWidth, 2):
            dCat[-1, j : j + 2] = centroids[dCat1D[idx]][0 : 2]
            idx += 1

    imageio.imwrite(outCat, dCat)
    print("Success on dequantization!")

if __name__ == "__main__":
    main(sys.argv[1:])
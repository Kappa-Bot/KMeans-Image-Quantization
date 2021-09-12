import numpy as np
import imageio

from sklearn.cluster import KMeans

from scipy.cluster.vq import vq, kmeans, whiten, kmeans2

import sys
from os.path import isfile
import struct

def main(argv):
    if len(argv) != 2:
        print("Usage: cat_quantizer.py <input_file> <output_file>")
        sys.exit(1)
    inCat, outCat = argv[0], argv[1]
    if not isfile(inCat):
        print("{} is not a valid file".format(inCat))
        sys.exit(2)
    print("Going to encode {} to output file {}".format(inCat, outCat))
    ###################################################################
    cat = np.ascontiguousarray(imageio.imread(inCat), dtype=np.uint8)
    (imHeight, imWidth) = (cat.shape[0], cat.shape[1])

    print("cat {} is {}x{} pixels".format(inCat, imHeight, imWidth))

    N = 4

    nPoints = (imWidth * imHeight) // N
    residualrows = imHeight % 2
    resres = imWidth * residualrows // N

    qCat = np.zeros([nPoints + resres, N], dtype = np.uint8)

    idx = 0
    for i in range(0, imHeight - residualrows - 1, 2):
        for j in range(0, imWidth, 2):
            qCat[idx][0 : N] = np.array(cat[i : i + 2, j : j + 2]).reshape([N])
            idx += 1

    if residualrows > 0:
        for j in range(0, imWidth, 2):
            qCat[idx][0 : N // 2] = qCat[idx][N // 2 : N] = \
                np.array(cat[-1, j : j + 2], ndmin = 1)
            idx += 1

    # Scipy Method
    centroids, qCat = kmeans2(np.array(qCat, dtype=np.float64),
                        k = N ** 2, iter = 300, thresh = 0.0001, minit = '++')
    centroids = np.array(centroids, dtype=np.uint8)
    qCat = np.array(qCat, dtype=np.uint8)

    # Sklearn Method
    #km = KMeans(n_clusters = N ** 2, init = 'k-means++', max_iter = 300, tol = 0.0001)
    #qCat = np.array(km.fit_predict(qCat), dtype=np.uint8)
    #centroids = np.array(km.cluster_centers_, dtype=np.uint8)

    auxDCat = np.ndarray(cat.shape, dtype=np.uint8)
    idx = 0
    for i in range(0, imHeight - residualrows - 1, 2):
        for j in range(0, imWidth, 2):
            auxDCat[i : i + 2, j : j + 2] = np.array(centroids[qCat[idx]][0 : N]).reshape([2, 2])
            idx += 1

    if residualrows > 0:
        for j in range(0, imWidth, 2):
            auxDCat[-1, j : j + 2] = centroids[qCat[idx]][0 : 2]
            idx += 1

    MSE = 0.0
    for i in range(imHeight):
        for j in range(imWidth):
            MSE += (float(auxDCat[i][j]) - float(cat[i][j])) ** 2
    MSE /= (imHeight * imWidth)
    print("MSE: {}".format(MSE))

    #err = np.sum((auxDCat.astype("float") - cat.astype("float")) ** 2)
    #err /= float(imWidth * imHeight)
    #print(err)

    # imageio.imwrite("{}.png".format(outCat), auxDCat) # For testing

    print("Creating file {}...".format(outCat), end='', flush=True)
    if not isfile(outCat):
        open(outCat, "x").close()
    f = open(outCat, "wb")
    print("done!")

    # 2B => dim + 1B => N
    print("Inserting headers.", end='', flush=True)
    f.write(struct.pack("i", imHeight))
    f.write(struct.pack("i", imWidth))
    f.write(struct.pack("B", N))
    print(".", end='', flush=True)

    # N * N ** 2 Bytes => Centroids
    for i in range(N ** 2):
        for j in range(N):
            f.write(centroids[i][j])
    print(".done!")

    print("Bit Packing", end = '', flush = True)
    residual = (nPoints + resres) % 2
    bitPack = np.zeros([(nPoints + resres) // 2 + residual], dtype = np.uint8) # 1D bitPacker
    """ The ideal mechanism is technically like this
        |  A  --  B  |
        | 0000  0000 |
        <--- byte --->
    """
    idx = 0
    for i in range(0, (nPoints + resres) - residual, 2):
        bitPack[idx] = ((qCat[i + 0] & 0b00001111) << 4) | \
                        (qCat[i + 1] & 0b00001111)
        idx += 1

    print(".", end='', flush=True)

    if residual > 0:  # Write 1 byte
        bitPack[idx] = ((qCat[(nPoints + resres) - residual] & 0b00001111) << 4)
        idx += 1
    print(".", end='', flush=True)

    for i in range((nPoints + resres) // 2 + residual):
        f.write(bitPack[i])
    print(".done!")

    print("Success on Quantization!")
    f.close()

if __name__ == "__main__":
    main(sys.argv[1:])

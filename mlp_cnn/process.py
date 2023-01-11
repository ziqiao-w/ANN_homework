import numpy as np
import matplotlib.pyplot as plt

meis = np.load("MEI/meis.npy")

mei0 = np.load("MEI/0.npy")

for index, value in enumerate(meis):
    # np.save("MEI/" + str(index), value)
    plt.axis("off")
    plt.imshow(value)
    plt.show()
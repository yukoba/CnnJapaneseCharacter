import numpy as np
# import scipy
import matplotlib.pyplot as plt

nb_classes = 72
img_rows, img_cols = 64, 64

ary = np.load("../src/hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15.0
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
# for i in range(nb_classes * 160):
#     X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
y_train = np.repeat(np.arange(nb_classes), 160)

plt.imshow(ary[71 * 160 + 1])
print(y_train[71 * 160 + 1])
plt.show()

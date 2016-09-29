import struct
import numpy as np
from PIL import Image

sz_record = 8199


def read_record_ETL8G(f):
    s = f.read(sz_record)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)


def read_hiragana():
    # Character type = 72, person = 160, y = 127, x = 128
    ary = np.zeros([72, 160, 127, 128], dtype=np.uint8)

    for j in range(1, 33):
        filename = '../ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                moji = 0
                for i in range(956):
                    r = read_record_ETL8G(f)
                    if b'.HIRA' in r[2]:
                        ary[moji, (j - 1) * 5 + id_dataset] = np.array(r[-1])
                        moji += 1
    np.savez_compressed("hiragana.npz", ary)

read_hiragana()


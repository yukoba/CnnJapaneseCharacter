# This code is from http://etlcdb.db.aist.go.jp/?page_id=2461

import struct
from PIL import Image

sz_record = 8199


def read_record_ETL8G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)


def test1():
    filename = '../ETL8G/ETL8G_01'
    id_record = 0

    with open(filename, 'rb') as f:
        f.seek(id_record * sz_record)
        r = read_record_ETL8G(f)

    print(r[0:-2], hex(r[1]))
    iE = Image.eval(r[-1], lambda x: 255 - x * 16)
    fn = '../tmp/ETL8G_{:d}_{:s}.png'.format((r[0] - 1) % 20 + 1, hex(r[1])[-4:])
    iE.save(fn, 'PNG')


def test2():
    filename = '../ETL8G/ETL8G_01'
    id_dataset = 0
    new_img = Image.new('L', (128 * 32, 128 * 30))

    with open(filename, 'rb') as f:
        f.seek(id_dataset * 956 * sz_record)
        for i in range(956):
            r = read_record_ETL8G(f)
            new_img.paste(r[-1], (128 * (i % 32), 128 * (i // 32)))
    iE = Image.eval(new_img, lambda x: 255 - x * 16)
    fn = '../tmp/ETL8G_ds{:03d}.png'.format(id_dataset)
    iE.save(fn, 'PNG')


def dump_all():
    for j in range(1, 33):
        for id_dataset in range(5):
            new_img = Image.new('L', (128 * 32, 128 * 30))

            filename = '../ETL8G/ETL8G_{:02d}'.format(j)
            with open(filename, 'rb') as f:
                f.seek(id_dataset * 956 * sz_record)
                for i in range(956):
                    r = read_record_ETL8G(f)
                    new_img.paste(r[-1], (128 * (i % 32), 128 * (i // 32)))
            iE = Image.eval(new_img, lambda x: 255 - x * 16)
            fn = '../tmp/ETL8G_ds{:02d}_{:01d}.png'.format(j, id_dataset)
            iE.save(fn, 'PNG')


test1()
test2()
# dump_all()

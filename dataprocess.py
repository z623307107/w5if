from cnn4 import *


def gen(data_filename):

    colorlist = [
        [0x00, 0x00, 0x00], # 黑
        [0xff, 0xff, 0xff], # 白
        [0xff, 0x00, 0x00], # 红
        [0x00, 0xff, 0x00], # 绿
        [0x00, 0x00, 0xff], # 蓝
        [0x00, 0xff, 0xff], # 黄
    ]
    anglelist = [135, 90, 45, 15]
    sizelist = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 17, 20]
    widthlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    data = []
    for c1 in colorlist:
        for c2 in colorlist:
            for c3 in colorlist:
                for a in anglelist:
                    for s in sizelist:
                        for w in widthlist:
                            data.append(c1 + c2 + c3 + [a, s, w])
    print(len(data))
    with open(data_filename, 'w') as file:
        for row in data:
            for col in row:
                file.write(str(col) + ' ')
            file.write('\n')

def load_model(model_name):
    return torch.load(io.BytesIO(open(path + 'net/' + model_name, 'rb').read()))

def use(model, labels, result_name):

    filestr = []

    with torch.no_grad():
        for label in torch.tensor(labels, dtype=torch.float32):
            output = model(label.unsqueeze(0))
            filestr.append((output.item(), ' '.join([str(int(i)) for i in label.tolist()])))

    filestr.sort(key=lambda x:x[0], reverse=True)

    with open(result_name, 'w') as file:
        for score, line in filestr:
            file.write(line + ' ' + str(int(score)) + '\n')

import sqlite3

def use2(model, labels, result_name):

    outputs = []

    with torch.no_grad():
        for label in torch.tensor(labels, dtype=torch.float32):
            outputs.append(int(model(label.unsqueeze(0)).item()))

    labels = [label + [output] for label, output in zip(labels, outputs)]

    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    c.execute(
        'create table xxx('
        'r1 int, g1 int, b1 int, '
        'r2 int, g2 int, b2 int, '
        'r3 int, g3 int, b3 int, '
        'angle int, size int, width int, '
        'score int)'
    )


    c.executemany('insert into xxx (r1, g1, b1, r2, g2, b2, r3, g3, b3, angle, size, width, score) values (?,?,?,?,?,?,?,?,?,?,?,?,?)', labels)
    conn.commit()


    print('color range:')
    c.execute('select distinct r1, g1, b1, r2, g2, b2, r3, g3, b3, score from xxx order by score desc limit 20')
    rows = c.fetchall()
    for row in rows:
        print(row)

    # 按角度排名
    print('algle range:')
    c.execute('select distinct angle, score from xxx order by score desc limit 20')
    rows = c.fetchall()
    for row in rows:
        print(row)

    print('size&width range:')
    c.execute('select distinct size, width, score from xxx order by score desc limit 20')
    rows = c.fetchall()
    for row in rows:
        print(row)

    conn.close()

if __name__ == '__main__':

    path = 'C:/Users/dawood/Desktop/CNN4/'


    mydata = []
    with open(path + 'mydata.txt', 'r') as file:
        mydata = [[int(i) if i.isdecimal() else int(i, 16) for i in line.split()][:12] for line in file]
    print('mydata read success')

    model_name = 'net019'
    model = load_model(model_name)
    print('model:', model_name)

    use2(model, mydata, path + 'result/' + model_name + '_result.txt')
    print('use ok')

    print('done!')


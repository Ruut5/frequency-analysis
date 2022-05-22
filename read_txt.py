import matplotlib.pyplot as plt

def read_txt(filename_):
    f = open(filename_,'r',encoding='UTF-8')
    txt = f.read()
    f.close
    return txt

def read_cast(filename_):
    f = open(filename_,'r',encoding='UTF-8')
    cod = f.readlines()
    chts = []
    abc =''
    for line in cod:
        abc+= str(line[0])
        chts+= float(line[2:-1])
    return abc,chts

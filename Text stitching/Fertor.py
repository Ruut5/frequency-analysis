ABC =   'ABCDEFGHIJKLMNOPQRSTUVWXYZ'\
        'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'\
        'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'\
        'abcdefghijklmnopqrstuvwxyz'\
        '0123456789'\
        ',.?!@#$%*:();:''""'

def get_ABC(key:str)-> str:
    abc = ABC
    for letter in key:
        if abc.find(letter) != -1:
            abc == abc.replace(letter, '')
    return key + abc

def clear_key(key) -> str:
    cl_key = ''
    for letter in key:
        if cl_key.find(letter) == -1:
            cl_key += letter
    return cl_key

def get_matrix(abc:str):
    mx = ['']*12
    mx = [ mx.copy() for i in mx ]
    mx = abs

def printMatr(a):
    for i in a:
        print(i)
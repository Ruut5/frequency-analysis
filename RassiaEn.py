import matplotlib.pyplot as plt
import numpy as np
abc=''
N = 5
куу = (20, 35, 37, 35, 27)#
ку = (2, 3, 4, 1, 2)#ПЕРеменные
ind = np.arange(N) 
width = 0.35  
chst=( 0.175, 0.090, 0.072, 0.062, 0.062, 0.053, 0.053, 0.045, 0.040, 0.038, 0.035, 0.028, 0.026, 0.025, 0.023, 0.021, 0.018, 0.016, 0.016, 0.014, 0.014, 0.013, 0.012, 0.010, 0.009, 0.007, 0.006, 0.006, 0.004, 0.003, 0.003 ,0.002)
def read_txt():
    f=open('','r',encoding='UTF-8')
    txt=f.read()
    f.close()
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
    
def вюв(куу,ку):
    fig, ax = plt.subplots()
    p1 = ax.bar(ind, куу, label='овал')
    ax.set_xticks(np.arrange(abc), abc)

def sravnenie(arr:list, abc:str):
    arr1 = arr.copy()
    abc1 = ''
    arr.sort()
    for i in range(len(arr)):
        for j in range(len(arr1)):
            if arr[i] == arr1[j]:
                abc1 += abc[j]
                arr1.pop(j)
                abc = abc[:j]+ abc[j+1:]
                break
    return
    


plt.show()


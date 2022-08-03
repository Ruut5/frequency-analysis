import loadnetMY
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
from PIL import ImageGrab
import numpy as np
from tkinter.ttk import Label
import matplotlib.pyplot as plt
import cv2
import pyautogui
import imutils
import os
'''
class new_Title_list(tk.Tk):
    def __init__(self):
        root.destroy()
        app = new_paint_panel()
        app.mainloop()
       
        tk.Tk.__init__(self)
        self.title("Титульный лист")
        self.geometry('700x800')
        self.configure(background = "black")

        path = r'titul.jpg'

        self.button_print = tk.Button(self, text = "Открыть окно программы", command = self.Next)
        self.button_print.pack(side="bottom", fill="both", expand=True)

        self.img = Image.open(r"titul.jpg")
        self.img = self.img.resize((600, 800), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.img)
        img = tk.Label(self,image = self.image)
        img.pack()
       

       
    def Next(self):
        root.destroy()
        app = new_paint_panel()
        app.mainloop()
       
'''

class new_paint_panel(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Распознавалка рукописных цифр")
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.brush_size = 10
        self.i_lable = 0
        self.lables = []

        def save():
            canvas = self._canvas()  # Координаты окна
            img = self.grabcanvas = ImageGrab.grab(bbox=canvas)
            #img = img.resize((28, 28), Image.ANTIALIAS)

            image = pyautogui.screenshot(region=(canvas)) # Скрин
            image = np.array(image) # Перевод в массив
            image = image[0:400,0:400] # Обрезка по панели
            imgdef = image # Бекап для работы с координатами
            imgray = cv2.cvtColor(imgdef, cv2.COLOR_BGR2GRAY) # Преобразование в формат для поиска образа
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Распознание образа
            print('\n'*4)
            if (len(contours) == 0):
                messagebox.showinfo("Результат", f"Нарисуйте хоть что-то \n")
            if (len(contours) > 1):  ## удаление лишних объектов (кругов в цифрах 0, 4, 6, 8, 9)
                delete = [] 
                for i in range(len(contours)):  ## поиск
                    for j in range(len(contours)): 
                        if (i != j):
                            x, y, w, h = cv2.boundingRect(contours[i])
                            x1, y1, w1, h1 = cv2.boundingRect(contours[j])
                            if (x1 > x and y1 > y) and (x1  < x + h and y1 < y+w):
                                delete.append(contours[j])
                for i in delete:              ## само удаление
                    contours.remove(i)
            # cntre = contours[0] # Берём координаты первого объекта
            for cntre in contours:  # проходим по всем найденым объектам
                x, y, w, h = cv2.boundingRect(cntre) # Получение координат угла + длинна и высота
                
                if (x < 50): # Корректировка координат для того
                    w +=x    # чтобы цифры были не слеплены c углом
                    x = 0
                else:
                    x -= 50
                    w += 50
                if (y < 50):
                    h += y
                    y = 0
                else:
                    y -= 50
                    h += 50
                
                if w > h :              # Преобразует прямоугольник в квадрат
                    h = w
                else:
                    w = h
                    
                if w % 28 != 0 :        # Преобразует квадрат в длимое 28-и 
                    s = w % 28          # Для избкжания неправильного масштабирование
                    w += (28 - s)
                    h = w

                img_arr = np.array(img)
                
                image_f = img_arr[y:y+h,x:x+w] # Обрезка
                    
                img_res = cv2.resize(image_f, (28, 28)) # масштабирование
                
                
                l = [[[[0]]*28]*28] # Перевод в другой вид массива
                l = np.array(l)
                l = l.astype('float32') 
                for i in range(28):
                    for j in range(28):
                        l[0][i][j][0] = (img_res[i][j][0]+img_res[i][j][1]+img_res[i][j][2])/3
                        
                img_reres = l
                
                plt.imshow(img_reres[0], cmap=plt.cm.binary) # Показ входных данных
                # plt.show()
                # '''                                               старая версия обработки изображения
                # img = img.convert('L')
                # plt.imshow(img, cmap=plt.cm.binary)
                # plt.show()
                
                # img_array = np.array(img.getdata()).reshape(1, 28, 28, 1)
                # '''
                img_array = np.array(img_reres) # Обработка в тензор
    
                img_array = img_array.astype('float32')
                img_array /= 255.0
                
                #print(img_array)
                arr = loadnetMY.model.predict(img_array) # Отправка данных в нейронку
                print(arr)
                g = 0
                for h in arr[0]:
                    g+=h
                print(g)   
                arr1 = np.argmax(arr)
                arr = arr.reshape(10,1)
                #print(arr)
                if max(arr) > 0.05: # Вывод результата
                    self.lables.append(tk.Label(self, text=f"Результат Вероятно вы нарисовали цифру: {arr1}"))
                    self.lables[self.i_lable].pack(side="top", fill="both", expand=True)
                    self.i_lable +=1
                    #messagebox.showinfo("Результат", "Вероятно вы нарисовали цифру: %.0f" % arr1)
                else:
                    messagebox.showinfo("Результат", f"Нарисуйте цифру \n{arr1}")
                plt.show()

        self.canv = tk.Canvas(self, width=400, height=400, bg = "black", cursor="cross")
        self.canv.pack(side="top", fill="both", expand=True)

        self.button_print = tk.Button(self, text = "Распознать", command = save)
        self.button_print.pack(side="top", fill="both", expand=True)
        

        self.button_clear = tk.Button(self, text = "Стереть", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)

        self.canv.bind("<Motion>", self.tell_me_where_you_are)
        self.canv.bind("<B1-Motion>", self.draw_from_where_you_are)

    

    def _canvas(self):
            #print('self.canv.winfo_rootx() = ', self.canv.winfo_rootx())
            #print('self.canv.winfo_rooty() = ', self.canv.winfo_rooty())
            #print('self.canv.winfo_x() =', self.canv.winfo_x())
            #print('self.canv.winfo_y() =', self.canv.winfo_y())
            #print('self.canv.winfo_width() =', self.canv.winfo_width())
            #print('self.canv.winfo_height() =', self.canv.winfo_height())
            x=self.canv.winfo_rootx()+self.canv.winfo_x()
            y=self.canv.winfo_rooty()+self.canv.winfo_y()
            x1=x+self.canv.winfo_width()
            y1=y+self.canv.winfo_height()
            box=(x+2,y+2,x1-2,y1-2)
            #print('box = ', box)
            return box


    def clear_all(self):
        self.canv.delete("all")
        for i in range(len(self.lables)):
            self.lables[i].destroy()
        self.i_lable = 0

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canv.create_oval(event.x - self.brush_size,
                          event.y - self.brush_size,
                          event.x + self.brush_size,
                          event.y + self.brush_size,
                          fill="#ffffff", outline="#ffffff")

        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)     
        self.points_recorded.append(self.x)        
        self.previous_x = self.x
        self.previous_y = self.y

if __name__ == "__main__":
    root = new_paint_panel()
    root.mainloop()
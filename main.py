import tkinter as tk 
import numpy as np 
from neural import *
 

class DigitWindow():
    def __init__(self, height=600, width=900):
        self.root = tk.Tk()
        self.HEIGHT = height
        self.WIDTH = width
        self.PIXEL = tk.PhotoImage(width=1, height=1)
        self.canvas = tk.Canvas(self.root, height=height, width=width, bd=0)
        self.pixels = [0 for _ in range(100)]
        self.entry = tk.Entry(self.canvas, width=5, font=40)
        self.digit_classifier = DigitClassifier()
        self.labels = {
            'static' : {
                'prediction' : tk.Label(
                    self.canvas, font=10, text='Prediction :'
                ),
                'confidence' : tk.Label(
                    self.canvas, font=10, text='Confidence :'
                )
            },
            'dynamic' : {
                'prediction' : tk.Label(
                    self.canvas, font=10, text=''
                ),
                'confidence' : tk.Label(
                    self.canvas, font=10, text=''
                )
            }
        }
        self.buttons = {
            'pixels' : [],
            'load-network' : tk.Button(
                self.canvas, image=self.PIXEL, text='Load Network', height=20, width=75, compound='c', 
                command=lambda:self.digit_classifier.load_network(input('Load Network from: '))
            ),
            'save-network' : tk.Button(
                self.canvas, image=self.PIXEL, text='Save Network', height=20, width=75, compound='c', 
                command=lambda:self.digit_classifier.save_network(input('Save Network to: '))
            ),
            'load-data' : tk.Button(
                self.canvas, image=self.PIXEL, text='Load Data', height=20, width=75, compound='c', 
                command=lambda:self.digit_classifier.load_data(input('Load dataset from: '))
            ),
            'save-data' : tk.Button(
                self.canvas, image=self.PIXEL, text='Save Data', height=20, width=75, compound='c', 
                command=lambda:self.digit_classifier.save_data(input('Save dataset to: '))
            ),
            'train' : tk.Button(
                self.canvas, image=self.PIXEL, text='Train', height=20, width=75, compound='c',
                command=lambda:self.digit_classifier.train()
            ),
            'predict' : tk.Button(
                self.canvas, image=self.PIXEL, text='Predict', height=20, width=75, compound='c',
                command=lambda:self.predict()
            ),
            'add-data' : tk.Button(
                self.canvas, image=self.PIXEL, text='Add Data', height=20, width=75, compound='c', 
                command=lambda:self.add_data()
            ),
            'shift-up' : tk.Button(
                self.canvas, image=self.PIXEL, text='Up', height=20, width=50, compound='c', 
                command=lambda:self.shift('up')
            ),
            'shift-left' : tk.Button(
                self.canvas, image=self.PIXEL, text='Left', height=20, width=50, compound='c', 
                command=lambda:self.shift('left')
            ),
            'shift-down' : tk.Button(
                self.canvas, image=self.PIXEL, text='Down', height=20, width=50, compound='c', 
                command=lambda:self.shift('down')
            ),
            'shift-right' : tk.Button(
                self.canvas, image=self.PIXEL, text='Right', height=20, width=50, compound='c', 
                command=lambda:self.shift('right')
            )
        }

 
    def pixels_click(self, index):
        '''
        change the color of the pixels button

        Parameter :
            index : index of the pixel -> Int
        '''
        if self.pixels[index] == 0:
            self.buttons['pixels'][index].config(bg="black")
            self.pixels[index] = 1
        else:
            self.buttons['pixels'][index].config(bg="white")
            self.pixels[index] = 0


    def predict(self):
        '''
        predict the digit using neural network
        '''
        prediction, confidence = self.digit_classifier.predict(np.array([1] + self.pixels))
        self.labels['dynamic']['prediction'].config(text=prediction)
        self.labels['dynamic']['confidence'].config(text=confidence * 100)


    def add_data(self):
        '''
        add training data for neural network
        '''
        try:
            y = int(self.entry.get()) 
            if 0 <= y < 10:
                y = [0 if y != _ else 1 for _ in range(10)]
                self.digit_classifier.add_data(np.array([[1] + self.pixels]), np.array([y]))
        except:
            print('Integer Expected!')


    def shift(self, direction):
        '''
        shift the pixel according to the direction

        Parameter :
            direction : the direction of the shift -> String
        '''
        if direction == 'left':
            for i in range(10):
                self.pixels.insert((i * 10) + 9, self.pixels.pop(i*10))

        elif direction == 'right':
            for i in range(10):
                self.pixels.insert(i*10,  self.pixels.pop((i * 10) + 9))

        elif direction == 'up':
            for i in range(10):
                self.pixels.append(self.pixels.pop(0))

        elif direction == 'down':
            for i in range(10):
                self.pixels.insert(0, self.pixels.pop())

        self.update()


    def update(self):
        '''
        show the updated vaue of the pixel box
        '''
        for i, pixel in enumerate(self.pixels):
            self.buttons['pixels'][i].config(bg="white" if pixel == 0 else "black")


    def draw(self):
        '''
        draw all the widget to the window
        '''
        self.canvas.pack()

        # Draw the pixel buttons
        for i in range(100):
            self.buttons['pixels'].append(tk.Button(
                self.canvas, image=self.PIXEL, height=50, width=50, bd=1, bg="white", 
                command=lambda index=i :self.pixels_click(index))
            )
            self.buttons['pixels'][i].place(x=(i % 10) * 50, y=(i // 10) * 50)
        
        # Shift Button
        self.buttons['shift-up'].place(x=100, y=510)
        self.buttons['shift-left'].place(x=25, y=550)
        self.buttons['shift-down'].place(x=100, y=550)
        self.buttons['shift-right'].place(x=175, y=550)

        # Other Button
        self.buttons['save-network'].place(x=520, y=440)
        self.buttons['load-network'].place(x=520, y=470)
        self.buttons['save-data'].place(x=620, y=440)
        self.buttons['load-data'].place(x=620, y=470)
        self.buttons['add-data'].place(x=520, y=500)
        self.buttons['predict'].place(x=520, y=530)
        self.buttons['train'].place(x=520, y=560)

        # The Labels
        self.labels['static']['prediction'].place(x=520, y=20)
        self.labels['static']['confidence'].place(x=520, y=80)
        self.labels['dynamic']['prediction'].place(x=550, y=50)
        self.labels['dynamic']['confidence'].place(x=550, y=110)
        self.entry.place(x=300, y=525)


    def start(self):
        '''
        start the window
        '''
        self.root.mainloop()


window = DigitWindow()
window.draw()
window.start()
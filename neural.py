import NeuralNetwork as nn 
import numpy as np 
import os

class DigitClassifier():
    def __init__(self, input_node=100):
        # Available data for training or validating
        self.data_x = np.array([])
        self.data_y = np.array([])

        # Training Configuration
        self.learning_rate = 0.1
        self.regularization_rate = 0.001
        self.batch_size = 50
        self.total_epoch = 5

        # Initialize the neural network
        self.network = nn.Network(input_node, self.learning_rate, self.regularization_rate, self.batch_size)

        # Hidden Layers
        self.network.add_layer(300)
        self.network.add_layer(100)

        # Output Layer
        self.network.add_layer(10)


    def load_data(self, filename):        
        '''
        Load the data from filename

        Parameter :
            filename : filename -> String
        '''
        try:
            self.data_x = np.load(f'{filename}_x.npy')
            self.data_y = np.load(f'{filename}_y.npy')
        except:
            print(f'Error : {filename} not found!')
    

    def save_data(self, filename):        
        '''
        Save the data to filename

        Parameter :
            filename : filename -> String
        '''
        np.save(f'{filename}_x.npy', self.data_x)
        np.save(f'{filename}_y.npy', self.data_y)


    def save_network(self, foldername):        
        '''
        Save the network weight into the foldername

        Parameter :
            foldername : foldername -> String
        '''
        self.network.save(foldername)
    

    def load_network(self, foldername):
        '''
        Load the network weight from the foldername

        Parameter :
            foldername : foldername -> String
        '''
        self.network.load(foldername)


    def add_data(self, x, y):        
        '''
        Add new data and result

        Parameter :
            x : new data -> Numpy Array
            y : new result -> Numpy Array
        '''
        if len(self.data_x) == 0:
            self.data_x = np.array(x)
            self.data_y = np.array(y)
        else:
            self.data_x = np.append(self.data_x, x, axis=0)
            self.data_y = np.append(self.data_y, y, axis=0)


    def shuffle(self):
        '''
        Shuffle the data in random order
        '''

        for index, swap in np.random.randint(0, len(self.data_x), (len(self.data_x), 2)):
            temp = np.copy(self.data_x[index])
            self.data_x[index] = np.copy(self.data_x[swap])
            self.data_x[swap] = temp

            temp = np.copy(self.data_y[index])
            self.data_y[index] = np.copy(self.data_y[swap])
            self.data_y[swap] = temp

        
    def train(self):       
        '''
        Train the network
        '''
        print('[Configuration]')
        print('Learning Rate       :', self.learning_rate)
        print('Regularization Rate :', self.regularization_rate)
        print('Batch Size          :', self.batch_size)
        print('Total Epochs        :', self.total_epoch)
        print('Start Training!')
        print('================')
        self.shuffle()
        self.network.train(self.data_x, self.data_y, self.total_epoch)
        print('================')
        print('Complete Training!')


    def predict(self, x):
        '''
        Get the last layer's activation for input data

        Parameter :
            x : input data -> Numpy Array

        return :
            the digit with highest confidence and the confidence -> Tuple
        '''
        output = self.network.predict(x)
        maximum = 0
        for i in range(1, len(output)):
            if output[i] > output[maximum]:
                maximum = i
        return (maximum, output[maximum])

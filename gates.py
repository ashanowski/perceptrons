import numpy as np


class Perceptron:

    def __init__(self, training_data, epochs=100, learning_rate=0.01):
        """
            Simple perceptron class

            training_data : list of numpy ndarrays
                First Numpy array contains training output (N x 2),
                second stores training input (1 x N).

            epochs : int
                Number of training cycles for perceptron

            learning_rate : float
                Step size for adjusting weights and bias

            weights : numpy ndarray
                M x 1 matrix of randomly generated weights
            
            bias : float
                Randomly generated bias
        """
        self.training_inputs = training_data[0]
        self.training_labels = training_data[1]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.random(len(self.training_inputs[0]))
        self.bias = np.random.random(1)

    def predict(self, inputs, bipolar=False):
        """ Predict activation """
        prediction = np.dot(inputs, self.weights) + self.bias
        if bipolar:
            if prediction > 0:
                return 1
            return - 1
        if not bipolar:
            if prediction > 0:
                return 1
            return 0

    def train(self):
        """ Train the model for epochs by adjusting weights and bias """
        for _ in range(self.epochs):
            for inputs, label in zip(self.training_inputs, self.training_labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)

    def print_equation(self):
        """ 
            Print the equation using weights and bias in the form of
                w1x1 + w2x2 + ... + b
        """
        for index, weight in enumerate(self.weights):
            print(weight, f'x{index + 1}', "+", end=' ')
        print(float(self.bias))


if __name__ == "__main__":

    training = {
        'and': [
            np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]),
            np.array([0, 0, 0, 1])
        ],
        'or': [
            np.array([
                [0, 0],
                [0, 1],
                [1, 0]
            ]),
            np.array([0, 1, 1])
        ],
        'not': [
            np.array([
                [1],
                [0]
            ]),
            np.array([0, 1])
        ],
        'xor': [
            np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ]),
            np.array([0, 1, 1, 0])
        ]
    }

    # perceptron = Perceptron(training['not'], learning_rate=0.01)
    # print('1 1 |', perceptron.predict([1, 1]))
    # print('1 0 |', perceptron.predict([1, 0]))
    # print('0 1 |', perceptron.predict([0, 1]))
    # print('0 0 |', perceptron.predict([0, 0]))

    perceptron = Perceptron(training['not'], learning_rate=0.01)
    print('0 |', perceptron.predict([0]))
    print('1 |', perceptron.predict([1]))

    perceptron.train()
    perceptron.print_equation()

    # print('1 1 |', perceptron.predict([1, 1]))
    # print('1 0 |', perceptron.predict([1, 0]))
    # print('0 1 |', perceptron.predict([0, 1]))
    # print('0 0 |', perceptron.predict([0, 0]))

    print('0 |', perceptron.predict([0]))
    print('1 |', perceptron.predict([1]))
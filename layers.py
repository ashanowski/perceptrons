import numpy as np
from tqdm import tqdm


class Layer:
    def __init__(self, no_neurons, no_inputs_per_neuron):
        self.weights = 2 * np.random.random((no_inputs_per_neuron, no_neurons)) - 1


class Network:
    def __init__(self, layer1: Layer, layer2: Layer):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        output_layer1 = self.__sigmoid(np.dot(inputs, self.layer1.weights))
        output_layer2 = self.__sigmoid(np.dot(output_layer1, self.layer2.weights))
        return output_layer1, output_layer2

    def train(self, training_input, training_output, epochs):
        for epoch in tqdm(range(epochs), desc="Training the model", unit='epoch'):

            output_layer1, output_layer2 = self.think(training_input)
            layer2_error = training_output - output_layer2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_layer2)

            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_layer1)

            layer1_adjustment = training_input.T.dot(layer1_delta)
            layer2_adjustment = output_layer1.T.dot(layer2_delta)

            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment

    def print_weights(self):
        print("---> Layer 1 (4 neurons, each with 3 inputs): ")
        print(self.layer1.weights)
        print("---------------------------------------------------")
        print("---> Layer 2 (1 neuron, with 4 inputs):")
        print(self.layer2.weights)
        print("---------------------------------------------------")
        print()

if __name__ == "__main__":
    # np.random.seed(1)

    # 4 neurons, 3 inputs each
    layer1 = Layer(4, 3)

    # 1 neuron, 4 inputs
    layer2 = Layer(1, 4)

    # network of 2 layers
    network = Network(layer1, layer2)

    training_inputs = np.array([
        # 1 at the end
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        # 0 at the end
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])

    training_outputs = np.array([[0, 1, 1, 0, 1, 1, 0]]).T

    print("===================================================")
    print("          Randomly generating weights...")
    print("===================================================")
    network.print_weights()

    network.train(training_inputs, training_outputs, 100000)

    print()
    print("===================================================")
    print("          New weights after training:")
    print("===================================================")
    network.print_weights()

    print("Test [1, 1, 0] situation:")

    _, output = network.think(np.array([1, 1, 0]))
    print(np.round(output))

import numpy as np

class NeuralNet:
    layerCount = 0
    shape = None
    weights = []

    def __init__(self, layerSize):
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        self._layerInput = []
        self._layerOutput = []

        for(l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(
                np.random.normal(scale=.1, size = (l2, l1 + 1))
            )

    def run(self, input):
        lnCases = input.shape[0]

        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                biasNodes = np.ones([1, lnCases])
                layerInput = self.weights[0].dot(np.vstack([input.T, biasNodes]))
            else:
                biasNodes = np.ones([1, lnCases])
                layerInput = self.weights[index].dot(
                    np.vstack(
                        [
                            self._layerOutput[-1],
                            biasNodes
                        ]
                    )
                )

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T

    def trainEpoch(self, input, target, trainingRate = 0.2):
        delta = []
        lnCases = input.shape[0]

        self.run(input)

        for index in reversed(range(self.layerCount)):
            if index ==self.layerCount - 1:
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(
                    output_delta * self.sgm(self._layerInput[index], True)
                )
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.sgm(self._layerInput[index], True))

        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
            else:
                bias_nodes = np.ones([1, self._layerOutput[index - 1].shape[1]])
                layerOutput = np.vstack([self._layerOutput[index - 1], bias_nodes])

            transposed_layer_output = layerOutput[None, :, :].transpose(2, 0, 1)
            transposed_deltas = delta[delta_index][None, :, :].transpose(2, 1, 0)

            print("===")
            print(layerOutput)
            print(delta[delta_index])
            print(transposed_layer_output)
            print(transposed_deltas)
            print(transposed_layer_output.shape)
            print(transposed_deltas.shape)
            print("===")
            weightDelta = np.sum(
                transposed_layer_output * transposed_deltas,
                axis = 0
            )

            self.weights[index] -= trainingRate * weightDelta

        return error

    def sgm(self, x, Derivative = False):
        if Derivative:
            out = self.sgm(x)
            return out * (1 - out)
        else:
            return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    nn = NeuralNet((2,2,1))
    print(nn.shape)
    print(nn.weights)

    lvInput = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    lvTarget = np.array([.05, .05, .95, .95])

    lnMax = 1
    lnErr = 1e-5
    for i in range(lnMax + 1):
        err = nn.trainEpoch(lvInput, lvTarget)
        if i % 10000 == 0:
            print("Iteratoin {0}\tError: {1:0.6f}".format(i, err))
        if err <= lnErr:
            print("Minimum error reached at iteration {0}".format(i))
            break

    lvOutput = nn.run(lvInput)
    print("Input: {0}\nOutput: {1}".format(lvInput, lvOutput))

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

class NeuralNetworkTest {

  [Test]
  public void NeuralNetwork_Run_ReturnsMatrixWithInputDimensions() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});
    const int inputRows = 4;
    const int inputColumns = 2;
    double[,] input = new double[inputRows, inputColumns] {
      { 1, 1 },
      { 0, 0 },
      { 1, 0 },
      { 0, 1 }
    };

    Matrix result = nn.Run(input);

    Assert.AreEqual(inputRows, result.RowCount);
    Assert.AreEqual(inputColumns, result.ColumnCount);
    // Our weights should start out really small so all outputs should end up
    // very close to .5 before any training has happened.
    IEnumerator<double> results = result.Enumerate().GetEnumerator();
    while(results.MoveNext()) {
      Assert.True(results.Current > .4);
      Assert.True(results.Current < .6);
    }
  }

  [Test]
  public void NeuralNetwork_Train_RunsSmoothly() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 2, 1});
    double[,] input = new double[,] {
      { 1, 1 },
      { 0, 0 },
      { 1, 0 },
      { 0, 1 }
    };

    double[,] target = new double[,] {
      {.95},
      {.95},
      {0.05},
      {0.05}
    };

    double error = nn.Train(input, target);

    Debug.Log(error);
  }

  [Test]
  public void NeuralNetwork_Sigmoid_ReturnsSigmoid() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});

    double result = nn.Sigmoid(0);

    Assert.AreEqual(.5, result);
  }

  [Test]
  public void NeuralNetwork_TransposeArrayTest() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});

    double[,] test = new double[,] {
      { 1, 1 },
      { 2, 2 },
      { 3, 3 }
    };

    double [,,] correct = new double[,,] {
      { {1, 1} },
      { {2, 2} },
      { {3, 3} },
    };

    Assert.AreEqual(correct, nn.DeltaTranspose(test));
  }

  [Test]
  public void NeuralNetwork_TransposeOutputArrayTest() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});

    double[,] test = new double[,] {
      { 1, 1 },
      { 2, 2 },
      { 3, 3 }
    };

    double [,,] correct = new double[,,] {
      { {1}, {1} },
      { {2}, {2} },
      { {3}, {3} },
    };

    Assert.AreEqual(correct, nn.OutputTranspose(test));
  }


  [Test]
  public void NeuralNetwork_SumAcrossZeroTest() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});

    double [,,] test = new double[,,] {
      { {1}, {1} },
      { {2}, {2} },
      { {3}, {3} },
    };
    double[,] correct = new double[,] {{6}, {6}};

    double[,] result = nn.SumAcrossZero(test);

    Assert.AreEqual(correct, result);
  }
}

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
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});
    double[,] input = new double[,] {
      { 1, 1 },
      { 0, 0 },
      { 1, 0 },
      { 0, 1 }
    };

    double[,] target = new double[,] {
      {1, 0},
      {1, 0},
      {0, 1},
      {0, 1}
    };

    nn.Train(input, target);
  }

  [Test]
  public void NeuralNetwork_Sigmoid_ReturnsSigmoid() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});

    double result = nn.Sigmoid(0);

    Assert.AreEqual(.5, result);
  }
}

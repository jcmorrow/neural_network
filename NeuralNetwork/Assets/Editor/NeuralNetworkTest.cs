using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

class NeuralNetworkTest {

  [Test]
  public void NeuralNetwork_Sigmoid_ReturnsSigmoid() {
    new NeuralNetwork(new List<int> {2, 3, 2});
  }

  [Test]
  public void NeuralNetwork_MatrixTesting_Test() {
    new NeuralNetwork(new List<int> {2, 3, 2}).MatrixTesting();
  }
}

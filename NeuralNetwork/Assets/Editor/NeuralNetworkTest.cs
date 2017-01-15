using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

class NeuralNetworkTest {

  [Test]
  public void NeuralNetwork_Run_CanReturnSomething() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});
    List<double> result = nn.Run(new List<double> { 1, 1 });
    foreach(double i in result) {
      Debug.Log(i);
    }
  }

  [Test]
  public void NeuralNetwork_Sigmoid_ReturnsSigmoid() {
    NeuralNetwork nn = new NeuralNetwork(new List<int> {2, 3, 2});

    // Debug.Log(nn.Sigmoid(-10));
  }
}

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

class BackPropogationNetworkTest {
  [Test]
  public void BackPropogationNetwork_Runs() {
    int[] layerSizes = new int[] { 2, 3, 1 };
    TransferFunction[] transferFunctions = new TransferFunction[] {
      TransferFunction.None,
      TransferFunction.Sigmoid,
      TransferFunction.Sigmoid
    };
    BackPropogationNetwork network = new BackPropogationNetwork(
      layerSizes,
      transferFunctions
    );
    double[] input = new double[] { 1.0, 1.0 };
    double[] output;

    network.Run(ref input, out output);

    foreach(double result in output) {
      Assert.True(result > 0);
      Assert.True(result < 1);
    }
  }

  [Test]
  public void BackPropogationNetwork_Trains() {
    int[] layerSizes = new int[] { 2, 3, 1 };
    TransferFunction[] transferFunctions = new TransferFunction[] {
      TransferFunction.None,
      TransferFunction.Sigmoid,
      TransferFunction.Sigmoid
    };
    BackPropogationNetwork network = new BackPropogationNetwork(
      layerSizes,
      transferFunctions
    );
    double[][] inputs = new double[][] {
      new double[] { 1.0, 1.0 },
      new double[] { 0.0, 0.0 },
      new double[] { 0.0, 1.0 },
      new double[] { 1.0, 0.0 },
    };
    double[][] targets = new double[][] {
      new double[] { 0.05 },
      new double[] { 0.05 },
      new double[] { 0.95 },
      new double[] { 0.95 },
    };
    double error = 0;

    for(int i = 0; i < 10000; i++) {
      for(int n = 0; n < inputs.Length; n++) {
        error = network.Train(ref inputs[n], ref targets[n], 0.15, 0.1);
      }
    }

    Debug.Log(error);
    Assert.AreEqual(0, error);
  }

  [Test]
  public void BackPropogationNetwork_Sigmoid_ReturnsLow() {
    // BackPropogationNetwork network = new BackPropogationNetwork();
  }
}

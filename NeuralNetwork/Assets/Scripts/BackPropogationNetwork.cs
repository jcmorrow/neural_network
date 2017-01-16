using System;
using UnityEngine;
using System.Collections.Generic;

public enum TransferFunction
{
  None,
  Sigmoid
}

static class TransferFunctions {
  public double Evaluate(TransferFunction tFunc, double input) {
    switch(tFunc)
    {
      case TransferFunction.Sigmoid:
        return sigmoid(input);
      case TransferFunction.None:
      default:
        return input;
    }
  }
  public double EvaluateDerivative(TransferFunction tFunc, double input) {
    switch(tFunc)
    {
      case TransferFunction.Sigmoid:
        return sigmoid_derivative(input);
      case TransferFunction.None:
      default:
        return 0.0;
    }
  }

  public double sigmoid(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }
  public double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
  }
}

public class BackPropogationNetwork{

  private int layerCount;
  private int inputSize;
  private int[] layerSize;
  private TransferFunction[] transferFunction;

  private double[][] layerOutput;
  private double[][] layerInput;
  private double[][] bias;
  private double[][] delta;
  private double[][] previousBiasDelta;

  private double[][][] weight;
  private double[][][] previousWeightDelta;

  public BackPropogationNetwork(int[] layerSizes, TransferFunction[] transferFunctions) {
    if(transferFunctions.Length != layerSizes.Length) {
      throw new ArgumentException("Fuck");
    }
    if(transferFunctions[0] != TransferFunctions.None) {
      throw new ArgumentException("Fuck");
    }

    layerCount = layerSizes.Length - 1;
    layerSize = new int[layerCount];

    for(int i = 0; i < layerCount; i++) {
      layerSize[i] = layerSizes[i + 1];
    }

    transferFunction = new TransferFunction(layerCount);

    for(int i = 0; i < layerCount; i++) {
      transferFunction[i] = transferFunctions[i + 1];
    }

    bias = new double[layerCount][];
    previousBiasDelta = new double[layerCount][];
    delta = new double[layerCount][];
    layerOutput = new double[layerCount][];
    layerInput = new double[layerCount][];

    weight = new double[layerCount][][];
    previousWeightDelta = new double[layerCount][][];

    for(int l = 0; l < layerCount; l++) {
      bias[l] = new double[layerSize[l]];
      previousBiasDelta[l] = new double[layerSize[l]];
      delta[l] = new double[layerSize[l]];
      layerOutput[l] = new double[layerSize[l]];
      layerInput[l] = new double[layerSize[l]];

      weight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];
      previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];

      for(int n = 0; n < (l == 0 ? inputSize : layerSize[l - 1]); n++) {
        weight[l][n] = new double[layerSize[l]];
        previousWeightDelta[l][n] = new double[layerSize[l]];
      }
    }
  }
}


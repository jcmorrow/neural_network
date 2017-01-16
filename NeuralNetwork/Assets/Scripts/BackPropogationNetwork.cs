using System;
using UnityEngine;
using System.Collections.Generic;

public enum TransferFunction
{
  None,
  Sigmoid
}

static class TransferFunctions {
  public static double Evaluate(TransferFunction tFunc, double input) {
    switch(tFunc)
    {
      case TransferFunction.Sigmoid:
        return sigmoid(input);
      case TransferFunction.None:
      default:
        return input;
    }
  }
  public static double EvaluateDerivative(TransferFunction tFunc, double input) {
    switch(tFunc)
    {
      case TransferFunction.Sigmoid:
        return sigmoid_derivative(input);
      case TransferFunction.None:
      default:
        return 0.0;
    }
  }

  public static double sigmoid(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }
  public static double sigmoid_derivative(double x) {
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
    if(transferFunctions[0] != TransferFunction.None) {
      throw new ArgumentException("Fuck");
    }

    layerCount = layerSizes.Length - 1;
    layerSize = new int[layerCount];

    for(int i = 0; i < layerCount; i++) {
      layerSize[i] = layerSizes[i + 1];
    }

    transferFunction = new TransferFunction[layerCount];

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

    for(int l = 0; l < layerCount; l++) {
      for(int j = 0; j < layerSize[l]; j++) {
        bias[l][j] = Gaussian.GetRandomGaussian();
        previousBiasDelta[l][j] = 0.0;
        layerOutput[l][j] = 0.0;
        layerInput[l][j] = 0.0;
        delta[l][j] = 0.0;
      }

      for(int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++) {
        for(int j = 0; j < layerSize[l]; j++) {
          weight[l][i][j] = Gaussian.GetRandomGaussian();
          previousWeightDelta[l][i][j] = 0.0;
        }
      }
    }
  }
}

public static class Gaussian {
  private static System.Random gen = new System.Random();

  public static double GetRandomGaussian() {
    return GetRandomGaussian(0.0, 1.0);
  }

  public static double GetRandomGaussian(double mean, double stddev) {
    double rVal1, rVal2;

    GetRandomGaussian(mean, stddev, out rVal1, out rVal2);

    return rVal1;
  }

  public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2) {
    double u, v, s, t;

    do{
      u = 2 * gen.NextDouble() - 1;
      v = 2 * gen.NextDouble() - 1;
    } while (u * u + v * v > 1 || (u == 0 && v == 0));

    s = u * u + v * v;
    t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

    val1 = stddev * u * t + mean;
    val2 = stddev * v * t + mean;
  }
}


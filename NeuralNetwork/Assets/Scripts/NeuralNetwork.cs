using System;
using UnityEngine;
using System.Collections.Generic;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
// using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;

public class NeuralNetwork {

  private int layerCount;
  private List<int> shape;
  private List<Matrix> weights;
  private List<Matrix> _layerInput;
  private List<Matrix> _layerOutput;

  public NeuralNetwork(List<int> layerShape) {
    layerCount = layerShape.Count - 1;
    shape = layerShape;
    weights = new List<Matrix>();

    int[] weightsArray1 = new int[shape.Count - 1];
    int[] weightsArray2 = new int[shape.Count - 1];

    shape.CopyTo(0, weightsArray1, 0, shape.Count - 1);
    shape.CopyTo(1, weightsArray2, 0, shape.Count - 1);

    for(int i = 0; i < weightsArray1.Length; i++) {
      weights.Add(
        Matrix.Build.Random(
          weightsArray2[i], weightsArray1[i] + 1
        ) * 0.01f
      );
    }
  }

  public Matrix Run(double[,] input) {
    Matrix inputMatrix = Matrix.Build.DenseOfArray(input);
    int numberOfCases = inputMatrix.RowCount;

    _layerInput = new List<Matrix>();
    _layerOutput = new List<Matrix>();

    for(int i = 0; i < layerCount; i++) {

      Matrix layerInput;
      Matrix layerOutput;

      if(i == 0) {
        Matrix biasNodes = Matrix.Build.Dense(1, numberOfCases, 1.0);
        layerInput = weights[i] * (inputMatrix.Transpose().Stack(biasNodes));
      } else {
        Matrix biasNodes = Matrix.Build.Dense(1, numberOfCases, 1.0);
        Matrix previousLayerOutputs = _layerOutput[_layerOutput.Count - 1];
        layerInput = weights[i] * previousLayerOutputs.Stack(biasNodes);
      }

      layerOutput = layerInput.Map(val => Sigmoid(val));

      _layerInput.Add(layerInput);
      _layerOutput.Add(layerOutput);
    }

    return _layerOutput[_layerOutput.Count - 1].Transpose();
  }

  public void Train(double[,] input, double[,] target, double trainingRate = .2) {
    List<Matrix> delta = new List<Matrix>();
    Matrix inputMatrix = Matrix.Build.DenseOfArray(input);
    Matrix targetMatrix = Matrix.Build.DenseOfArray(target);
    int numberOfCases = inputMatrix.RowCount;

    Run(input);

    for(int index = layerCount - 1; index >= 0; index--) {
      if(index == layerCount - 1) {
        Matrix outputDelta = _layerOutput[index] - targetMatrix.Transpose();
        // double error = (outputDelta.PointwisePower(2)).RowSums().Sum();
        // I'm really not sure about this Transpose
        Matrix sigmoid = _layerInput[index].Map(val => Sigmoid(val, true)).Transpose();
        delta.Add(outputDelta * sigmoid);
      } else {
        Matrix lastDelta = delta[delta.Count - 1];
        Matrix deltaPullback = weights[index + 1].Transpose() * lastDelta;
        Matrix withoutBiases = deltaPullback.SubMatrix(
          0, deltaPullback.RowCount - 1,
          0, deltaPullback.ColumnCount
        );
        Matrix sigmoid = _layerInput[index].Map(val => Sigmoid(val, true));
        // I'm really not sure about this Transpose
        delta.Add(withoutBiases.Transpose() * sigmoid);
      }
    }
  }

  public double Sigmoid(double x, bool derivative = false) {
    if(derivative) {
      double output = Sigmoid(x);
      return output * (1 - output);
    } else {
      return 1 / (1 + Math.Exp(-x));
    }
  }
}

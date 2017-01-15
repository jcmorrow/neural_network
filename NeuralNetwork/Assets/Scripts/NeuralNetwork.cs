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

  public double Train(double[,] input, double[,] target, double trainingRate = .2) {
    double error = 0;
    List<Matrix> delta = new List<Matrix>();
    Matrix inputMatrix = Matrix.Build.DenseOfArray(input);
    Matrix targetMatrix = Matrix.Build.DenseOfArray(target);
    int numberOfCases = inputMatrix.RowCount;

    Run(input);

    // compute deltas
    for(int index = layerCount - 1; index >= 0; index--) {
      if(index == layerCount - 1) {
        Matrix outputDelta = _layerOutput[index] - targetMatrix.Transpose();
        error = (outputDelta.PointwisePower(2)).RowSums().Sum();
        Matrix sigmoid = _layerInput[index].Map(val => Sigmoid(val, true));
        delta.Add(outputDelta.PointwiseMultiply(sigmoid));
      } else {
        Matrix lastDelta = delta[delta.Count - 1];
        Matrix deltaPullback = weights[index + 1].Transpose() * lastDelta;
        Matrix withoutBiases = deltaPullback.SubMatrix(
          0, deltaPullback.RowCount - 1,
          0, deltaPullback.ColumnCount
        );
        Matrix sigmoid = _layerInput[index].Map(val => Sigmoid(val, true));
        delta.Add(withoutBiases.PointwiseMultiply(sigmoid));
      }
    }

    // weight deltas - otherwise known as the tough part
    for(int index = 0; index < layerCount; index++) {
      int deltaIndex = layerCount - 1 - index;

      Matrix layerOutput;

      if(index == 0){
        Matrix biasNodes = Matrix.Build.Dense(1, numberOfCases, 1.0);
        layerOutput = inputMatrix.Transpose().Stack(biasNodes);
      } else {
        Matrix previousLayerOutputs = _layerOutput[index - 1];
        Matrix biasNodes = Matrix.Build.Dense(1, previousLayerOutputs.ColumnCount, 1.0);
        layerOutput = previousLayerOutputs.Stack(biasNodes);
      }

      double[,] layerOutputArray = layerOutput.ToArray();
      double[,,] layerOutputOtherThing = new double[,,] {{{}}};

      Debug.Log(layerOutput);

      // np.sum doesn't exist. What we want to do is sum these across the 0th
      // axis
       weightDelta = np.sum(
         OutputTranspose(layerOutput) * DeltaTranspose(delta[deltaIndex])
       );
    }

    return error;
  }

  public double[,,] DeltaTranspose(double[,] toTranspose) {
    double[,,] result = new double[toTranspose.GetLength(0), 1, toTranspose.GetLength(1)];
    for(int row = 0; row < toTranspose.GetLength(0); row++) {
      for(int column = 0; column < toTranspose.GetLength(1); column++) {
        Debug.Log(toTranspose[row, column]);
        result[row, 0, column] = toTranspose[row, column];
      }
    }

    return result;
  }

  public double[,,] OutputTranspose(double[,] toTranspose) {
    double[,,] result = new double[toTranspose.GetLength(0), toTranspose.GetLength(1), 1];
    for(int row = 0; row < toTranspose.GetLength(0); row++) {
      for(int column = 0; column < toTranspose.GetLength(1); column++) {
        Debug.Log(toTranspose[row, column]);
        result[row, column, 0] = toTranspose[row, column];
      }
    }

    return result;
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

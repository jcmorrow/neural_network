using System;
using UnityEngine;
using System.Collections.Generic;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;

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
        ) * 0.1f
      );
    }
  }

  public List<double> Run(List<double> input) {
    Vector inputVector = Vector.Build.DenseOfEnumerable(input);
    int inputsCount = input.Count;

    _layerInput = new List<Matrix>();
    _layerOutput = new List<Matrix>();

    Matrix layerInput;
    for(int i = 0; i < layerCount; i++) {
      if(i == 0) {
        Matrix columnMatrix = inputVector.ToRowMatrix();
        Matrix biasNode = Matrix.Build.Dense(1, 1, 1.0);
        layerInput = weights[0] * columnMatrix.Append(biasNode).Transpose();
      } else {
        Matrix columnMatrix = _layerOutput[_layerOutput.Count - 1].Transpose();
        Matrix biasNode = Matrix.Build.Dense(1, 1, 1.0);
        layerInput = weights[i] * columnMatrix.Append(biasNode).Transpose();
      }

      Matrix layerOutput = Matrix.Build.Dense(layerInput.RowCount, layerInput.ColumnCount);
      layerInput.CopyTo(layerOutput);
      IEnumerator<Vector> layerInputValues = (IEnumerator<Vector>)layerInput.EnumerateRows();
      int row = 0;
      while(layerInputValues.MoveNext()) {
        Vector vec = layerInputValues.Current;
        for(int col = 0;col < vec.Count; col++) {
          layerOutput[row, col] = Sigmoid(layerInput[row, col]);
        }
      }

      _layerInput.Add(layerInput);
      _layerOutput.Add(layerOutput);
    }

    return new List<double>(
      _layerOutput[_layerOutput.Count - 1].Enumerate()
    );
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

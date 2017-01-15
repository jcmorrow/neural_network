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

  public void MatrixTesting() {
    Vector inputVector = Vector.Build.DenseOfEnumerable(
      new List<double> { 1.0, 1.0 }
    );

    Matrix matrix = Matrix.Build.DenseOfMatrixArray(
      new Matrix[,] {{
        inputVector.ToColumnMatrix(),
        Matrix.Build.Dense(1, 2, 1.0)
      }}
    );
    Debug.Log(inputVector.ToColumnMatrix());
    Debug.Log(Matrix.Build.Dense(1, 2, 1.0));
    Debug.Log(matrix);
  }

  public void Run(List<double> input) {
    Vector inputVector = Vector.Build.DenseOfEnumerable(input);
    int inputsCount = input.Count;

    for(int i = 0; i < layerCount; i++) {
      if(i == 0) {
        Matrix layerInput = weights[0] * Matrix.Build.DenseOfMatrixArray(
          new Matrix[,] {{
            inputVector.ToColumnMatrix(),
            Matrix.Build.Dense(1, inputsCount, 1.0)
          }}
        );
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

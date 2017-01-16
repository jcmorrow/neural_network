using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

class BackPropogationNetworkTest {
  [Test]
  public void BackPropogationNetwork_Init_Runs() {
    BackPropogationNetwork bpn = new BackPropogationNetwork();
  }

  [Test]
  public void BackPropogationNetwork_Sigmoid_ReturnsLow() {
    BackPropogationNetwork network = new BackPropogationNetwork();
  }
}

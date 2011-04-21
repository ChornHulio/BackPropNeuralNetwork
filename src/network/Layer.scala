package network

class Layer(cLength: Int, nLength : Int) {
  private val length = cLength
  private val nextLength = nLength // length of the next layer
  private val neurons = new Array[Neuron](length)

  private var learningRate = 1

  for (i <- 0 until length) {
    neurons(i) = new Neuron(i, nextLength)
  }

  def getCurrentValues() = {
    val currentValues = new Array[Double](length)
    for (i <- 0 until length) {
      currentValues(i) = neurons(i).getCurrentValue
    }
    currentValues
  }

  def getNeurons() = neurons
  def getNeuron(index: Int) = {
    if (index < neurons.length) {
      neurons(index)
    } else {
      null
    }
  }

  def setDeltas(delta : Array[Double]) {
    for (
      i <- 0 until length if i < delta.length
    ) {
      neurons(i).setDelta(delta(i))
    }
  }

  def getDeltas() {
    val deltas = new Array[Double](length)
    for (i <- 0 until length) {
      deltas(i) = neurons(i).getDelta()
    }
    deltas
  }

  def setWeights(index: Int, weights: Array[Double]) {
    for (
      i <- 0 until length if i < weights.length; if index < length
    ) {
      neurons(index).setWeightTo(i, weights(i))
    }
  }

  def feed(in: Array[Double]) {
    for (
      i <- 0 until length if i < in.length
    ) {
      neurons(i).setCurrentValue(in(i))
    }
  }

  def learn(delta: Array[Double]) { 
    for (i <- 0 until length) {
      for (j <- 0 until delta.length) {
        val newWeight = neurons(i).getWeightTo(j) + neurons(i).getCurrentValue * delta(j) * learningRate
        neurons(i).setWeightTo(j, newWeight)
      }
    }
    learningRate = learningRate * 100 / 101
  }
}
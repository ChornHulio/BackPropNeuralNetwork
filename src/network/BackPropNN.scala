package network

import scala.util.Random

class BackPropNN(inNeuronsCount: Int, outNeuronsCount: Int,
  hiddenNeuronsCount: Int, hiddenLayersCount: Int) {

  private val inputLayer = new Layer(inNeuronsCount, hiddenNeuronsCount)
  private val outputLayer = new Layer(outNeuronsCount, 0)
  private val hiddenLayer = new Array[Layer](hiddenLayersCount)

  for (i <- 0 until hiddenLayersCount) {
    hiddenLayer(i) = {
      if (i < hiddenLayersCount - 1) new Layer(hiddenNeuronsCount, hiddenNeuronsCount)
      else new Layer(hiddenNeuronsCount, outNeuronsCount)
    }
  }

  /**
   * Train the network with the array of TrainingData count times
   *
   * @param in Array of TrainingData
   * @param count How often should the network be trained
   */
  def train(in: Array[TrainingData], count: Int) {
    for (i <- 0 until count) {
      val random = new Random()
      trainOnce(in(random.nextInt(in.length)))
    }
  }

  /**
   * Execute the network with the given data
   *
   * @param in InputData to execute
   */
  def execute(data: ExecuteData) = {
    // feed input layer
    inputLayer.feed(data.getIn())

    // give information through the network (feedforwarding)
    feedforwarding()

    // return output
    outputLayer.getNeurons()
  }

  def printWeightGraph() {
    println("--------------weights--------------")
    println("Input-Layer:")
    for (i <- 0 until inputLayer.getNeurons.length) {
      print("\t" + i + ": ")
      val neurons = inputLayer.getNeurons
      for (j <- 0 until hiddenLayer(0).getNeurons.length) {
        print(neurons(i).getWeightTo(j) + " - ")
      }
      print("\n")
    }

    println("Hidden-Layer:")
    for (i <- 0 until hiddenLayer(0).getNeurons.length) {
      print("\t" + i + ": ")
      val neurons = hiddenLayer(0).getNeurons
      for (j <- 0 until outputLayer.getNeurons.length) {
        print(neurons(i).getWeightTo(j) + " - ")
      }
      print("\n")
    }
  }

  def printCurrentValueGraph() {
    println("--------current values-----------")
    print("Input-Layer: ")
    val iLNeurons = inputLayer.getNeurons
    for (i <- 0 until iLNeurons.length) {
      print(iLNeurons(i).getCurrentValue + " - ")
    }
    println("")
    for (j <- 0 until hiddenLayer.length) {
      print("Hidden-Layer " + j + ": ")
      val neurons = hiddenLayer(j).getNeurons
      for (i <- 0 until iLNeurons.length) {
        print(neurons(i).getCurrentValue + " - ")
      }
      println("")
    }
    print("Output-Layer: ")
    val oLNeurons = outputLayer.getNeurons
    for (i <- 0 until oLNeurons.length) {
      print(oLNeurons(i).getCurrentValue + " - ")
    }
    println("")
  }

  def printDeltaGraph() {
    println("--------------deltas---------------")
    print("Input-Layer: ")
    val iLNeurons = inputLayer.getNeurons
    for (i <- 0 until iLNeurons.length) {
      print(iLNeurons(i).getDelta + " - ")
    }
    println("")
    for (j <- 0 until hiddenLayer.length) {
      print("Hidden-Layer " + j + ": ")
      val neurons = hiddenLayer(j).getNeurons
      for (i <- 0 until iLNeurons.length) {
        print(neurons(i).getDelta + " - ")
      }
      println("")
    }
    print("Output-Layer: ")
    val oLNeurons = outputLayer.getNeurons
    for (i <- 0 until oLNeurons.length) {
      print(oLNeurons(i).getDelta + " - ")
    }
    println("")
  }

  /**
   * Train the network with one TrainingData
   *
   * @param data The data for training
   */
  private def trainOnce(data: TrainingData) {
    // feed input layer
    inputLayer.feed(data.getIn())

    // give information through the network (feedforwarding)
    feedforwarding()

    // calculate delta (backpropagation)
    backpropagation(data)

    // learn    
    learn()
  }

  private def feedforwarding() {
    /**
     * notes:
     *  pNode = previousNode
     *  cNode = currentNode
     *  nNode = nextNode
     *  --> previous- / current- / next-layer-Neurons are meant
     */
    for (i <- 0 to hiddenLayer.length) {
      val pNodes = {
        if (i == 0) inputLayer.getNeurons
        else hiddenLayer(i - 1).getNeurons
      }
      val layerLength = {
        if (i < hiddenLayer.length) hiddenLayer(i).getNeurons.length
        else outputLayer.getNeurons.length
      }
      val data = new Array[Double](layerLength)
      for (k <- 0 until layerLength) {
        data(k) = 0
        for (j <- 0 until pNodes.length) {
          data(k) += pNodes(j).getCurrentValue * pNodes(j).getWeightTo(k)
        }
        // scale it
        data(k) = 1 / (1 + math.exp(-data(k)))
      }
      if (i < hiddenLayer.length) hiddenLayer(i).feed(data)
      else outputLayer.feed(data)
    }
  }

  private def backpropagation(data: TrainingData) {
    var i = hiddenLayer.length
    while (i >= 0) {
      val cNodes = {
        if (i == hiddenLayer.length) outputLayer.getNeurons
        else hiddenLayer(i).getNeurons
      }
      val nNodes = {
        if (i == hiddenLayer.length) null
        else if (i == hiddenLayer.length - 1) outputLayer.getNeurons
        else hiddenLayer(i + 1).getNeurons
      }
      val delta = new Array[Double](cNodes.length)
      for (j <- 0 until cNodes.length) {
        val errFactor = {
          if (i == hiddenLayer.length) {
            data.getOut(j) - cNodes(j).getCurrentValue
          } else {
            var sum = 0.0
            for (k <- 0 until nNodes.length) {
              sum += nNodes(k).getDelta * cNodes(i).getWeightTo(k)
            }
            sum
          }
        }
        delta(j) = cNodes(j).getCurrentValue * (1 - cNodes(j).getCurrentValue) * errFactor
      }
      if (i == hiddenLayer.length) outputLayer.setDeltas(delta)
      else hiddenLayer(i).setDeltas(delta)
      i -= 1
    }
  }

  private def learn() {
    for (i <- 0 to hiddenLayer.length) {
      val nNodes = {
        if (i == hiddenLayer.length) outputLayer.getNeurons
        else hiddenLayer(i).getNeurons
      }

      val deltas = new Array[Double](nNodes.length)
      for (i <- 0 until nNodes.length) {
        deltas(i) = nNodes(i).getDelta()
      }

      if (i == 0) inputLayer.learn(deltas)
      else hiddenLayer(i - 1).learn(deltas)
    }
  }

}
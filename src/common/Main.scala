package common
import network._

object Main {

  val size = 10

  def main(args: Array[String]): Unit = {
		  
	// init
    val network = new BackPropNN(2,1,2,1)
    
    // training
    val trainingData = new Array[TrainingData](4)
	trainingData(0) = new TrainingData(Array(0.0, 0.0), Array(0.0), "1")
    trainingData(1) = new TrainingData(Array(0.0, 1.0), Array(1.0), "2")
    trainingData(2) = new TrainingData(Array(1.0, 0.0), Array(1.0), "3")
    trainingData(3) = new TrainingData(Array(1.0, 1.0), Array(0.0), "4")
    network.train(trainingData, 10)
    
    // executing
    val executeDataA = new ExecuteData(Array(0.0, 0.0), "1")
    print("\n1: ")
    network.execute(executeDataA).foreach((x : Neuron) => print(x.getCurrentValue + "\t"))
    
    val executeDataB = new ExecuteData(Array(0.0, 1.0), "2")
    print("\n2: ")
    network.execute(executeDataB).foreach((x : Neuron) => print(x.getCurrentValue + "\t"))
    
    val executeDataC = new ExecuteData(Array(1.0, 0.0), "3")
    print("\n3: ")
    network.execute(executeDataC).foreach((x : Neuron) => print(x.getCurrentValue + "\t"))
    
    val executeDataD = new ExecuteData(Array(1.0, 1.0), "4")
    print("\n4: ")
    network.execute(executeDataD).foreach((x : Neuron) => print(x.getCurrentValue + "\t"))
  }
}
package network

import scala.util.Random

class Neuron(i : Int, length : Int) {
	private val index = i;
	private val weightTo = new Array[Double](length)
	private var currentValue = 0.0
	private var delta = 0.0
	
	for(i <- 0 until length) {
		val random = new Random()
		weightTo(i) = (random.nextDouble() - 0.5) * 1000
	}
	
	def setWeightTo(index : Int, weight : Double) {
		weightTo(index) = weight;
	}
	
	def getWeightTo(index : Int) = {
		if(index < weightTo.length) {
			weightTo(index)
		} else {
			0
		}
	}
	
	def setCurrentValue(x : Double) {currentValue = x}	
	def getCurrentValue() = currentValue
	
	def setDelta(x : Double) {delta = x}	
	def getDelta() = delta
}
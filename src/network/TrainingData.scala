package network

class TrainingData(in : Array[Double], out :Array[Double], str : String) {
	private val input = in
	private val output = out
	private val string = str
	
	def getIn() = input
	
	def getOut() = output
	def getOut(x : Int) = {
		if(x < output.length) {
			output(x)
		} else {
			0
		}
	}
	
	def getString() = string
}
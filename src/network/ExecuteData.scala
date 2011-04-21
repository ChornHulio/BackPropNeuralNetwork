package network

class ExecuteData(in : Array[Double], str : String) {
	private val input = in
	private val string = str
	
	def getIn() = input
	def getString() = string
}
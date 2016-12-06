package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.LinearLayer;
import duyuNN.NNInterface;
import duyuNN.SoftmaxLayer;

public class LinearSoftmax implements NNInterface{

	public LinearLayer linear;
	public SoftmaxLayer softmax;
	
	public int inputLength;
	public int outputLength;
	
	public int linkId;
	
	public LinearSoftmax()
	{
	}
	
	public LinearSoftmax(int xInputLength, int xOutputLength) 
			throws Exception
	{
		inputLength = xInputLength;
		outputLength = xOutputLength;
		
		linear = new LinearLayer(inputLength, outputLength);
		softmax = new SoftmaxLayer(outputLength);
		
		linear.link(softmax);
		
		linkId = 0;
	}
	
	public void forward()
	{
		linear.forward();
		softmax.forward();
	}
	
	public void backward()
	{
		softmax.backward();
		linear.backward();
	}
	
	public void update(double learningRate)
	{
		linear.update(learningRate);
	}
	
	public void clearGrad()
	{
		linear.clearGrad();
		softmax.clearGrad();
	}
	
	public Object cloneWithTiedParams()
	{
		LinearSoftmax clone = new LinearSoftmax();
		
		clone.inputLength = inputLength;
		clone.outputLength = outputLength;
		clone.linkId = 0;
		
		clone.linear = (LinearLayer)linear.cloneWithTiedParams();
		clone.softmax = (SoftmaxLayer)softmax.cloneWithTiedParams();
		
		try {
			clone.linear.link(clone.softmax);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != softmax.output.length || nextIG.length != softmax.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		softmax.output = nextI;
		softmax.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		link(nextLayer, linkId);
	}
	
	public Object getInput(int id) {
		return linear.input;
	}

	public Object getOutput(int id) {
		return linear.output;
	}

	public Object getInputG(int id) {
		return linear.inputG;
	}

	public Object getOutputG(int id) {
		return linear.outputG;
	}

	@Override
	public void randomize(Random r, double min, double max) {
		linear.randomize(r, min, max);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}
}

package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.LinearLayer;
import duyuNN.NNInterface;
import duyuNN.TanhLayer;

public class LinearTanh implements NNInterface{

	public LinearLayer linear;
	public TanhLayer tanh;
	
	public int inputLength;
	public int outputLength;
	
	public int linkId;
	
	public LinearTanh()
	{
	}
	
	public LinearTanh(int xInputLength, int xOutputLength) 
			throws Exception
	{
		inputLength = xInputLength;
		outputLength = xOutputLength;
		
		linear = new LinearLayer(inputLength, outputLength);
		tanh = new TanhLayer(outputLength);
		
		linear.link(tanh);
		
		linkId = 0;
	}
	
//	public void randomize()
//	{
//		Random rnd = new Random();
//		linear.randomize(rnd, -0.01 / linear.inputLength, 0.01 / linear.inputLength);
//	}
	
	public void forward()
	{
		linear.forward();
		tanh.forward();
	}
	
	public void backward()
	{
		tanh.backward();
		linear.backward();
	}
	
	public void update(double learningRate)
	{
		linear.update(learningRate);
	}
	
	public void clearGrad()
	{
		linear.clearGrad();
		tanh.clearGrad();
	}
	
	public LinearTanh cloneWithTiedParams()
	{
		LinearTanh clone = new LinearTanh();
		
		clone.inputLength = inputLength;
		clone.outputLength = outputLength;
		clone.linkId = 0;
		
		clone.linear = (LinearLayer)linear.cloneWithTiedParams();
		clone.tanh = (TanhLayer)tanh.cloneWithTiedParams();
		
		try {
			clone.linear.link(clone.tanh);
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
		
		if(nextI.length != tanh.output.length || nextIG.length != tanh.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		tanh.output = nextI;
		tanh.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		link(nextLayer, linkId);
	}
	
	public Object getInput(int id) {
		return linear.input;
	}

	public Object getOutput(int id) {
		return tanh.output;
	}

	public Object getInputG(int id) {
		return linear.inputG;
	}

	public Object getOutputG(int id) {
		return tanh.outputG;
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

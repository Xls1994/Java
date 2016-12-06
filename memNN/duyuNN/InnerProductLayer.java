package duyuNN;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

public class InnerProductLayer implements NNInterface{

	public int eachLength;
	
	double[] input;
	double[] inputG;
	
	public double[] output;
	public double[] outputG;
	
	public void dumpToStream(Writer bw) throws IOException
    {
    	bw.write(eachLength + "\n");
    }
    
    public static InnerProductLayer loadFromStream(BufferedReader br) throws IOException 
    {
    	int _length = Integer.parseInt(br.readLine());
    	InnerProductLayer layer = new InnerProductLayer(_length);
    	return layer;
    }
	
	public InnerProductLayer(int xEachInputLength) {
		eachLength = xEachInputLength;
		
		input = new double[eachLength * 2];
		inputG = new double[eachLength * 2];
		output = new double[1];
		outputG = new double[1];
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		
	}

	@Override
	public void forward() {
		
		output[0] = 0;
		for(int i = 0; i < eachLength; i++)
		{
			output[0] += input[i] * input[eachLength + i];
		}
	}

	@Override
	public void backward() {
		for(int i = 0; i < eachLength; i++)
		{
			inputG[i] += outputG[0] * input[eachLength + i];
			inputG[eachLength + i] += outputG[0] * input[i];
		}
	}

	@Override
	public void update(double learningRate) {
		
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < inputG.length; i++)
		{
			inputG[i] = 0;
		}
		outputG[0] = 0;
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != output.length || nextIG.length != outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		output = nextI;
		outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, 0);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return input;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		InnerProductLayer inner = new InnerProductLayer(eachLength);
		
		return inner;
	}

}

package duyuNN;

import java.util.Random;

public class ElementMultiLayer implements NNInterface{

	public int length;

 	public double[] output;
 	public double[] outputG;

 	public double[] leftInput;
 	public double[] rightInput;
 	public double[] leftInputG;
 	public double[] rightInputG;
 
 	public int linkId;
    
 	public ElementMultiLayer()
 	{
 	}
 	
 	public ElementMultiLayer(int xElementLength)
 	{
 		this(xElementLength, 0);
 	}
 	
 	public ElementMultiLayer(int xElementLength, int xLinkId)
 	{
 		length = xElementLength;
 		linkId = xLinkId;
 		
 		leftInput = new double[length];
 		leftInputG = new double[length];
 		rightInput = new double[length];
 		rightInputG = new double[length];
 		output = new double[length];
 		outputG = new double[length];
 	}
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		for(int i = 0; i < length; i++)
		{
			output[i] = leftInput[i] * rightInput[i];
		}
	}

	@Override
	public void backward() 
	{
		// TODO Auto-generated method stub
		for (int i = 0; i < length; i++)
        {
            leftInputG[i] = outputG[i] * rightInput[i];
        }

        for (int i = 0; i < length; i++)
        {
            rightInputG[i] = outputG[i] * leftInput[i];
        }
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputG.length; i++)
		{
			outputG[i] = 0;
		}
		
		for(int i = 0; i < length; i++)
		{
			leftInputG[i] = 0;
		}
		
		for(int i = 0; i < length; i++)
		{
			rightInputG[i] = 0;
		}
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
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		if(id == 0)
		{
			return leftInput;
		}
		else if(id == 1)
		{
			return rightInput;
		}
		else
		{
			return null;
		}
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		if(id == 0)
		{
			return leftInputG;
		}
		else if(id == 1)
		{
			return rightInputG;
		}
		else
		{
			return null;
		}
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		ElementMultiLayer clone = new ElementMultiLayer(length, linkId);
		return clone;
	}
}

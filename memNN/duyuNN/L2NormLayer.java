package duyuNN;

import java.util.Random;

public class L2NormLayer implements NNInterface{

	public int length;
	public int linkId;
	public double[] input;
	public double[] inputG;
	public double[] output;
	public double[] outputG;
	
	public L2NormLayer()
	{
	}
	
	public L2NormLayer(int xLength)
	{
		this(xLength, 0);
	}
	
	public L2NormLayer(int xLength, int xLinkId)
	{
		length = xLength;
		linkId = xLinkId;
		input = new double[length];
		inputG = new double[length];
		output = new double[length];
		outputG = new double[length];
	}
	
	private double L2Norm(double[] x)
    {
        double sum = 0;

        for (int i = 0; i < x.length; ++i)
        {
            sum += x[i] * x[i];
        }

        double norm = Math.sqrt(sum);

        return norm;
    }
	
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		double z = L2Norm(input);

        if (z <= 0)
        {
        	for (int i = 0; i < input.length; ++i)
            {
                output[i] = input[i]; 
            }
            return;
        }

        for (int i = 0; i < input.length; ++i)
        {
            output[i] = input[i] / z; 
        }
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		double z = L2Norm(input);

        if (z <= 0)
        {
            return;
        }
        
        for (int i = 0; i < input.length; i++)
        {
            for (int j = 0; j < output.length; j++)
            {
                if (i == j)
                {
                    inputG[i] += outputG[j] * (1 + -1.0 * output[i] * output[i]) / z;
                }
                else
                {
                    inputG[i] += outputG[j] * -1.0 * output[i] * output[j] / z;
                }
            }
        }
        
//        for (int i = 0; i < input.length; ++i)
//        {
//            inputG[i] = outputG[i] / z;
//            for (int j = 0; j < input.length; ++j)
//            {
//                inputG[i] += -outputG[j] * (output[i] * output[j]) / z;
//            }
//        }
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
		
		for(int i = 0; i < inputG.length; i++)
		{
			inputG[i] = 0;
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
		SoftmaxLayer clone = new SoftmaxLayer(length, linkId); 
		return clone;
	}

}

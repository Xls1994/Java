package duyuNN;

import java.util.Random;

public class BiLinearLayer implements NNInterface{

	private LinearLayer[] WRs;
    private LinearLayer VL;
    private LinearLayer VR;
    
    public int outputLength;
    public int leftLength;
    public int rightLength;

    public double[][] input;
    public double[] output;
    public double[][] inputG;
    public double[] outputG;

    public int linkId;
	
    private BiLinearLayer()
    {
    }
    
    public BiLinearLayer(int xLeftLength, 
    		int xRightLength,
    		int xOutputLength)
    {
    	this(xLeftLength, xRightLength, xOutputLength, 0);
    }
    
    public BiLinearLayer(int xLeftLength, 
    		int xRightLength, 
    		int xOutputLength,
    		int xLinkId)
    {
    	leftLength = xLeftLength;
    	linkId = xLinkId;
    	rightLength = xRightLength;
    	outputLength = xOutputLength;
    	
    	for(int i = 0; i < outputLength; i++)
    	{
    		WRs[i] = new LinearLayer(rightLength, leftLength);
    	}
    	
    	VL = new LinearLayer(leftLength, outputLength);
    	VR = new LinearLayer(rightLength, outputLength);
    	
    	input = new double[2][];
    	input[0] = new double[leftLength];
    	input[1] = new double[rightLength];
    	
    	inputG = new double[2][];
    	inputG[0] = new double[leftLength];
    	inputG[1] = new double[rightLength];
    	
    	output = new double[outputLength];
    	outputG = new double[outputLength];
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for (int i = 0; i < WRs.length; ++i)
        {
            WRs[i].randomize(r, min, max);
        }

        VL.randomize(r, min, max);
        VR.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		for (int i = 0; i < WRs.length; ++i)
        {
            WRs[i].input = input[1];
            WRs[i].forward();
            output[i] = MathOp.dotProduct(WRs[i].output, input[0]);
        }
		
		VL.input = input[0];
		VR.input = input[1];
		
		VL.forward();
		VR.forward();
		
		for(int i = 0; i < output.length; i++)
		{
			output[i] += VL.output[i];
			output[i] += VR.output[i];
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		VL.outputG = outputG;
        VL.inputG = inputG[0];
        VL.backward();
        
        VR.outputG = outputG;
        VR.inputG = inputG[1];
        VR.backward();
        
        for (int i = 0; i < WRs.length; ++i)
        {
        	for(int j = 0; j < outputLength; j++)
        	{
        		WRs[i].outputG[j] += outputG[j] * input[0][j];
        	}
        	WRs[i].backward();
        }
        
        for (int i = 0; i < WRs.length; ++i)
        {
        	for (int j = 0; j < input[0].length; ++j)
            {
                inputG[0][j] += outputG[i] * WRs[i].output[j];
            }
        	
        	for (int j = 0; j < inputG[1].length; ++j)
            {
                inputG[1][j] += WRs[i].inputG[j];
            }
        }
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputLength; i++)
		{
			WRs[i].update(learningRate);
		}
		VL.update(learningRate);
		VR.update(learningRate);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputLength; i++)
		{
			WRs[i].updateAdaGrad(learningRate, batchsize);
		}
		VL.updateAdaGrad(learningRate, batchsize);
		VR.updateAdaGrad(learningRate, batchsize);
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputLength; i++)
		{
			WRs[i].clearGrad();
		}
		VL.clearGrad();
		VR.clearGrad();
		
		for(int i = 0; i < outputLength; i++)
		{
			outputG[i] = 0;
		}
		for(int i = 0; i < inputG.length; i++)
		{
			for(int j = 0; j < inputG[i].length; j++)
			{
				inputG[i][j] = 0;
			}
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
		return input[id];
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG[id];
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		BiLinearLayer clone = new BiLinearLayer();
		
		clone.rightLength = rightLength;
		clone.leftLength = leftLength;
		clone.outputLength = outputLength;
		clone.linkId = linkId;
		
		clone.input = new double[2][];
		clone.input[0] = new double[leftLength];
		clone.input[1] = new double[rightLength];
		clone.inputG = new double[2][];
		clone.inputG[0] = new double[leftLength];
		clone.inputG[1] = new double[rightLength];
		clone.output = new double[outputLength];
		clone.outputG = new double[outputLength];
		
		clone.VL = (LinearLayer) VL.cloneWithTiedParams();
		clone.VR = (LinearLayer) VR.cloneWithTiedParams();
		clone.WRs = new LinearLayer[WRs.length];
		for(int i = 0; i < outputLength; i++)
		{
			clone.WRs[i] = (LinearLayer) WRs[i].cloneWithTiedParams();
		}
		return clone;
	}

}

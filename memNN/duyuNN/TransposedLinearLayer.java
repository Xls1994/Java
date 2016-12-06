package duyuNN;

import java.util.Random;

public class TransposedLinearLayer implements NNInterface{

	// LinearLayer is y = Wx + b
	// TransposedLinearLayer is y = xW + b
	// Useful for AutoencoderLayer and LowRankBiLinearLayer
	
	public double[][] W;
    public double[] b;

    public int inputLength;
    public int outputLength;

    public double[] input;
    public double[] output;

    public double[] outputG;
    public double[] inputG;

    public double[][] WG;
    public double[] bG;
	
    public double[][] WAdaLR;
    public double[] bAdaLR;
	
    int linkId;
    
    private TransposedLinearLayer()
    {
    }

    public TransposedLinearLayer(int xInputLength, int xOutputLength)
    {
    	this(xInputLength, xOutputLength, 0);
    }
    
    public TransposedLinearLayer(int xInputLength, 
    		int xOutputLength, int xLinkId)
    {
    	linkId = xLinkId;
    	inputLength = xInputLength;
    	outputLength = xOutputLength;
    	
    	input = new double[inputLength];
    	inputG = new double[inputLength];
    	output = new double[outputLength];
    	outputG = new double[outputLength];
    	
    	b = new double[outputLength];
    	bG = new double[outputLength];
    	bAdaLR = new double[outputLength];
    	
    	W = new double[inputLength][];
    	WG = new double[inputLength][];
    	WAdaLR = new double[inputLength][];
    	
    	for(int i = 0; i < W.length; i++)
    	{
    		W[i] = new double[outputLength];
    		WG[i] = new double[outputLength];
    		WAdaLR[i] = new double[outputLength];
    	}
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				W[i][j] = r.nextFloat() * (max - min) + min;
			}
		}
		for(int i = 0; i < b.length; i++)
		{
			b[i] = r.nextFloat() * (max - min) + min;
		}
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		MathOp.xDotApb(input, W, b, output);
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		for(int i = 0; i < output.length; i++)
		{
			bG[i] += outputG[i];
		}
		for(int i = 0; i < inputG.length; i++)
		{
			inputG[i] = 0;
		}
		MathOp.Axpy(W, outputG, inputG);
		
		MathOp.A_add_xTmulty(input, outputG, WG);
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		for(int i = 0; i < bG.length; i++)
		{
			b[i] += learningRate * bG[i];
		}
		
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				W[i][j] += learningRate * WG[i][j];
			}
		}
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		for(int i = 0; i < b.length; i++)
		{
			bAdaLR[i] += (bG[i] / batchsize) * (bG[i] / batchsize);
			b[i] += learningRate / batchsize * bG[i] / Math.sqrt(bAdaLR[i]);
		}
		
		for(int i = 0; i < W.length; i++)
		{
			for(int j = 0; j < W[i].length; j++)
			{
				WAdaLR[i][j] += (WG[i][j] / batchsize) * (WG[i][j] / batchsize);
				W[i][j] += (learningRate / batchsize) * WG[i][j] / Math.sqrt(WAdaLR[i][j]);
			}
		}
	}

	@Override
	public void clearGrad()
	{
		// TODO Auto-generated method stub
		for (int i = 0; i < WG.length; ++i)
        {
			for(int j = 0; j < WG[i].length; j++)
			{
				WG[i][j] = 0;
			}
        }

        for (int i = 0; i < bG.length; ++i)
        {
            bG[i] = 0;
        }

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
		TransposedLinearLayer clone = new TransposedLinearLayer();
		
		clone.linkId = linkId;
		clone.inputLength = inputLength;
		clone.outputLength = outputLength;
    	
		clone.input = new double[inputLength];
		clone.inputG = new double[inputLength];
		clone.output = new double[outputLength];
		clone.outputG = new double[outputLength];
    	
		clone.b = b;
		clone.bG = new double[outputLength];
		clone.bAdaLR = bAdaLR;
    	
		clone.W = W;
		clone.WG = new double[inputLength][];
		clone.WAdaLR = WAdaLR;
		
		for(int i = 0; i < W.length; i++)
    	{
    		WG[i] = new double[outputLength];
    	}
		
		return clone;
	}
}

package duyuNN;

import java.util.Random;

public class LowRankBiLinearLayer implements NNInterface{

	private LinearLayer[] WRs;
    private TransposedLinearLayer[] LWs;
    private LinearLayer VL;
    private LinearLayer VR;

    private double[][] diagG;
    private double[][] diagAda;

    public int[] inputLength;
    public int outputLength;
    public double[][] diag;

    public int leftLength;
    public int rightLength;
    public int rank;

    public double[][] input;
    public double[] output;
    public double[][] inputG;
    public double[] outputG;

    public int linkId;

    private LowRankBiLinearLayer()
    {
    }

    public LowRankBiLinearLayer(int xLeftLength, 
    		int xRightLength, int xRank, int xOutputLength)
    {
    	this(xLeftLength, xRightLength, xRank, xOutputLength, 0);
    }
    
    public LowRankBiLinearLayer(int xLeftLength, 
    		int xRightLength, int xRank, int xOutputLength, int xLinkId)
    {
    	leftLength = xLeftLength;
    	rightLength = xRightLength;
    	rank = xRank;
    	outputLength = xOutputLength;
    	linkId = xLinkId;
    	
    	input = new double[2][];
    	input[0] = new double[leftLength];
    	input[1] = new double[rightLength];
    	
    	inputG = new double[2][];
    	inputG[0] = new double[leftLength];
    	inputG[1] = new double[rightLength];
    	
    	output = new double[outputLength];
    	outputG = new double[outputLength];
    	
    	WRs = new LinearLayer[outputLength];
    	LWs = new TransposedLinearLayer[outputLength];
    	for(int i = 0; i < outputLength; i++)
    	{
    		WRs[i] = new LinearLayer(rightLength, rank);
    		LWs[i] = new TransposedLinearLayer(leftLength, rank);
    	}
    	VL = new LinearLayer(xLeftLength, outputLength);
    	VR = new LinearLayer(rightLength, outputLength);
    	
    	for(int i = 0; i < outputLength; i++)
    	{
    		diag[i] = new double[Math.min(leftLength, rightLength)];
    		diagG[i] = new double[Math.min(leftLength, rightLength)];
    		diagAda[i] = new double[Math.min(leftLength, rightLength)];
    	}
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputLength; i++)
		{
			WRs[i].randomize(r, min, max);
			LWs[i].randomize(r, min, max);
		}
		VL.randomize(r, min, max);
		VR.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputLength; i++)
		{
			WRs[i].input = input[1];
			WRs[i].forward();
			
			LWs[i].input = input[0];
			LWs[i].forward();
			
			output[i] = MathOp.dotProduct(LWs[i].output, WRs[i].output);
		}
		
		VL.input = input[0];
		VL.forward();
		for (int i = 0; i < output.length; ++i)
        {
            output[i] += VL.output[i];
        }
		
		VR.input = input[1];
		VR.forward();
		for (int i = 0; i < output.length; ++i)
        {
            output[i] += VR.output[i];
        }
		
		for(int i = 0; i < outputLength; i++)
		{
			double x = 0;
			for(int j = 0; j < diag[0].length; j++)
			{
				x += input[0][j] * diag[i][j] * input[1][j];
			}
			output[i] += x;
		}
	}

	@Override
	public void backward() 
	{
		// TODO Auto-generated method stub
		VL.outputG = outputG;
        VL.inputG = inputG[0];
        VL.backward();
        
        VR.outputG = outputG;
        VR.inputG = inputG[1];
        VR.backward();
        
        for (int i = 0; i < outputG.length; ++i)
        {
            for (int j = 0; j < rank; ++j)
            {
                LWs[i].outputG[j] = outputG[i] * WRs[i].output[j];
                WRs[i].outputG[j] = outputG[i] * LWs[i].output[j];
            }
            LWs[i].backward();
            WRs[i].backward();
            for (int j = 0; j < inputG[0].length; ++j)
            {
                inputG[0][j] += LWs[i].inputG[j];
            }
            for (int j = 0; j < inputG[1].length; ++j)
            {
                inputG[1][j] += WRs[i].inputG[j];
            }
        }
        
        for (int i = 0; i < output.length; ++i)
        {
            double og = outputG[i];
            for (int j = 0; j < diag[i].length; ++j)
            {
                inputG[0][j] += og * diag[i][j] * input[1][j];
                inputG[1][j] += og * diag[i][j] * input[0][j];
                diagG[i][j]  += og * input[0][j] * input[1][j];
            }
        }
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		for (int i = 0; i < LWs.length; ++i)
        {
            LWs[i].update(learningRate);
        }
        for (int i = 0; i < WRs.length; ++i)
        {
            WRs[i].update(learningRate);
        }

        VL.update(learningRate);
        VR.update(learningRate);

        for (int i = 0; i < diagG.length; ++i)
        {
            for (int j = 0; j < diagG[i].length; ++j)
            {
                diag[i][j] += learningRate * diagG[j][j];
            }
        }
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		for (int i = 0; i < LWs.length; ++i)
        {
            LWs[i].updateAdaGrad(learningRate, batchsize);
        }
        for (int i = 0; i < WRs.length; ++i)
        {
            WRs[i].updateAdaGrad(learningRate, batchsize);
        }

        VL.updateAdaGrad(learningRate, batchsize);
        VR.updateAdaGrad(learningRate, batchsize);
        
        for (int i = 0; i < diagG.length; ++i)
        {
            for (int j = 0; j < diagG[i].length; ++j)
            {
            	diagAda[i][j] += (diagG[i][j] / batchsize) 
            			* (diagG[i][j] / batchsize);
            	
                diag[i][j] += learningRate / batchsize 
                		* diagG[i][j] / Math.sqrt(diagAda[i][j]);
            }
        }
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for (int i = 0; i < LWs.length; ++i)
        {
            LWs[i].clearGrad();
        }
        for (int i = 0; i < WRs.length; ++i)
        {
            WRs[i].clearGrad();
        }

        VL.clearGrad();
        VR.clearGrad();
        
        for(int i = 0; i < inputG.length; i++)
        {
        	for(int j = 0; j < inputG[i].length; j++)
        	{
        		inputG[i][j] = 0;
        	}
        }
        for(int i = 0; i < outputG.length; i++)
        {
        	outputG[i] = 0;
        }
        for(int i = 0; i < diagG.length; i++)
        {
        	for(int j = 0; j < diagG[i].length; j++)
        	{
        		diagG[i][j] = 0;
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
		LowRankBiLinearLayer clone = new LowRankBiLinearLayer();
		
		clone.linkId = linkId;
		clone.leftLength = leftLength;
		clone.rightLength = rightLength;
		clone.rank = rank;
		clone.outputLength = outputLength;
		
		clone.input = new double[2][];
		clone.input[0] = new double[leftLength];
		clone.input[1] = new double[rightLength];
    	
		clone.inputG = new double[2][];
		clone.inputG[0] = new double[leftLength];
		clone.inputG[1] = new double[rightLength];
    	
		clone.output = new double[outputLength];
		clone.outputG = new double[outputLength];
		
		clone.WRs = new LinearLayer[outputLength];
		for(int i = 0; i < WRs.length; i++)
		{
			clone.WRs[i] = (LinearLayer) WRs[i].cloneWithTiedParams();
		}
		
		clone.LWs = new TransposedLinearLayer[outputLength];
		for(int i = 0; i < LWs.length; i++)
		{
			clone.LWs[i] = (TransposedLinearLayer) LWs[i].cloneWithTiedParams();
		}
		
		clone.VL = (LinearLayer) VL.cloneWithTiedParams();
		clone.VR = (LinearLayer) VR.cloneWithTiedParams();
		
		clone.diag = diag;
		clone.diagAda = diagAda;
		clone.diagG = new double[outputLength][];
		for(int i = 0; i < diagG.length; i++)
		{
			clone.diagG[i] = new double[diagG[i].length];
		}
		return clone;
	}
}

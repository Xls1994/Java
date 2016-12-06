package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.LinearLayer;
import duyuNN.LookupLayer;
import duyuNN.NNInterface;
import duyuNN.TanhLayer;

public class LookupLinearTanhLinear implements NNInterface{
	
	public int windowSize;
	public int vocabSize;
	public int hiddenLength;
	public int embeddingLength;
	public int outputLength;
	
	public LookupLayer lookup;
	public LinearLayer linear;
	public TanhLayer tanh;
	public LinearLayer linear2;
	
	int linkId;
	
	public LookupLinearTanhLinear()
	{
	}
	
	public LookupLinearTanhLinear(LookupLayer seedLookup,
			LinearLayer seedLinear,
			LinearLayer seedLinear2) throws Exception
	{
		vocabSize = seedLookup.vocabSize;
		hiddenLength = seedLinear.outputLength;
		embeddingLength = seedLookup.embeddingLength;
		windowSize = seedLookup.inputLength;
		
		lookup = (LookupLayer) seedLookup.cloneWithTiedParams();
		linear = (LinearLayer) seedLinear.cloneWithTiedParams();
		tanh = new TanhLayer(hiddenLength);
		linear2 = (LinearLayer) seedLinear2.cloneWithTiedParams();
		
		lookup.link(linear);
		linear.link(tanh);
		tanh.link(linear2);
	}
	
	public LookupLinearTanhLinear(
		int xWindowSize,
		int xVocabSize,
		int xHiddenLength,
		int xOutputLength,
		int xEmbeddingLength) 
	{
		vocabSize = xVocabSize;
		hiddenLength = xHiddenLength;
		embeddingLength = xEmbeddingLength;
		windowSize = xWindowSize;
		outputLength = xOutputLength;
		
		lookup = new LookupLayer(embeddingLength, vocabSize, windowSize);
		linear = new LinearLayer(windowSize * embeddingLength, hiddenLength);
		tanh = new TanhLayer(hiddenLength);
		linear2 = new LinearLayer(hiddenLength, outputLength);
		
		try {
			lookup.link(linear);
			linear.link(tanh);
			tanh.link(linear2);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void forward()
	{
		lookup.forward();
		linear.forward();
		tanh.forward();
		linear2.forward();
	}
	
	public void backward()
	{
		linear2.backward();
		tanh.backward();
		linear.backward();
		lookup.backward();
	}
	
	public void clearGrad()
	{
		lookup.clearGrad();
		linear.clearGrad();
		tanh.clearGrad();
		linear2.clearGrad();
	}
	
	public LookupLinearTanhLinear cloneWithTiedParams() 
	{
		LookupLinearTanhLinear clone = new LookupLinearTanhLinear();
		
		clone.vocabSize = vocabSize;
		clone.hiddenLength = hiddenLength;
		clone.embeddingLength = embeddingLength;
		clone.windowSize = windowSize;
		clone.linkId = linkId;
		clone.outputLength = outputLength;
		
		clone.lookup = (LookupLayer) lookup.cloneWithTiedParams();
		clone.linear = (LinearLayer) linear.cloneWithTiedParams();
		clone.tanh = (TanhLayer) tanh.cloneWithTiedParams();
		clone.linear2 = (LinearLayer) linear2.cloneWithTiedParams();
		
		try {
			clone.lookup.link(clone.linear);
			clone.linear.link(clone.tanh);
			clone.tanh.link(clone.linear2);
			
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
		
		if(nextI.length != linear2.output.length || nextIG.length != linear2.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		linear2.output = nextI;
		linear2.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		link(nextLayer, linkId);
	}
	
	public Object getInput(int id) {
		return null;
	}

	public Object getOutput(int id) {
		return linear2.output;
	}

	public Object getOutputG(int id) {
		return linear2.outputG;
	}

	public void randomizeFanIn(Random r, double min, double max) {
		linear.randomize(r, min/linear.inputLength, max/linear.inputLength);
		linear2.randomize(r, min/linear2.inputLength, max/linear2.inputLength);
		lookup.randomize(r, min, max);
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		linear.randomize(r, min, max);
		linear2.randomize(r, min, max);
		lookup.randomize(r, min, max);
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(double learningRate) {
		linear.update(learningRate);
		linear2.update(learningRate);
		lookup.update(learningRate);
	}
	
	public void updateFanIn(double learningRate)
	{
		linear.update(learningRate / linear.inputLength);
		linear2.update(learningRate / linear2.inputLength);
		lookup.update(learningRate);
	}
	
}

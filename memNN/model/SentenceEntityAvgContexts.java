package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import duyuNN.AverageLayer;
import duyuNN.LinearLayer;
import duyuNN.LookupLayer;
import duyuNN.MultiConnectLayer;
import duyuNN.NNInterface;
import duyuNN.SoftmaxLayer;
import duyuNN.TanhLayer;
import duyuNN.combinedLayer.AttentionLayer;
import duyuNN.combinedLayer.ConnectLinearTanh;
import duyuNN.combinedLayer.ConnectLinearTanhLinear;
import duyuNN.combinedLayer.LookupLinearTanh;
import duyuNN.combinedLayer.SimplifiedLSTMLayer;

public class SentenceEntityAvgContexts implements NNInterface{
	
	List<LookupLayer> lookupList;
	MultiConnectLayer lookupConnect;
	
	AverageLayer averageLayer;
	
	int hiddenLength;
	int linkId;
	
	double[] targetVec;
	
	public double[] output;
	public double[] outputG;

	public SentenceEntityAvgContexts()
	{
		
	}
	
	public SentenceEntityAvgContexts(
			int[] wordIds,
			LookupLayer seedLookup,
			double[] xTargetVec
		) throws Exception 
	{
		targetVec = new double[xTargetVec.length];
		System.arraycopy(xTargetVec, 0, targetVec, 0, xTargetVec.length);
		
		hiddenLength = seedLookup.embeddingLength;
		lookupList = new ArrayList<LookupLayer>();
		
		int[] lookupConnectLengths = new int[wordIds.length];
		Arrays.fill(lookupConnectLengths, hiddenLength);
		lookupConnect = new MultiConnectLayer(lookupConnectLengths);
		
		for(int i = 0; i < wordIds.length; i++)
		{
			LookupLayer tmpLookup = (LookupLayer) seedLookup.cloneWithTiedParams();
			tmpLookup.input[0] = wordIds[i];
			tmpLookup.link(lookupConnect, i);
			
			lookupList.add(tmpLookup);
		}
		
		averageLayer = new AverageLayer(lookupConnect.outputLength, hiddenLength);
		
		lookupConnect.link(averageLayer);
		
		output = new double[hiddenLength];
		outputG = new double[hiddenLength];
		
		linkId = 0;
	}
	
	
	
	@Override
	public void randomize(Random r, double min, double max) {
		
	}

	@Override
	public void forward() {
		for(int i = 0; i < lookupList.size(); i++)
		{
			lookupList.get(i).forward();
		}
		
		lookupConnect.forward();
		
		averageLayer.forward();
		
		for(int i = 0; i < output.length; i++)
		{
			output[i] = averageLayer.output[i] + targetVec[i];
		}
	}

	@Override
	public void backward() {
		
//		for(int i = 0; i < outputG.length; i++)
//		{
//			averageLayer.outputG[i] = outputG[i];
//		}
//		
//		averageLayer.backward();
//		
//		lookupConnect.backward();
//		
//		for(int j = 0; j < lookupList.size(); j++)
//		{
//			lookupList.get(j).backward();
//		}
	}

	@Override
	public void update(double learningRate) {
		
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
	}

	@Override
	public void clearGrad() {
		
		for(int i = lookupList.size() - 1; i >= 0 ; i--)
		{
			lookupList.get(i).clearGrad();
		}
		lookupList.clear();
		
		lookupConnect.clearGrad();
		
		averageLayer.clearGrad();
		
		Arrays.fill(output, 0);
		Arrays.fill(outputG, 0);
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != output.length 
				|| nextIG.length != outputG.length)
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
		return null;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		return null;
	}
}

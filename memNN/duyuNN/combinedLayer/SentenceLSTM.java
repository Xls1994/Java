package duyuNN.combinedLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import duyuNN.AverageLayer;
import duyuNN.LookupLayer;
import duyuNN.MultiConnectLayer;
import duyuNN.NNInterface;
import duyuNN.TanhLayer;
import duyuNN.combinedLayer.LookupLinearTanh;
import duyuNN.combinedLayer.SimplifiedLSTMLayer;

public class SentenceLSTM implements NNInterface{
	
	// output is tanhList.get(tanhList.size() - 1)
	
	List<LookupLayer> lookupList;
	List<SimplifiedLSTMLayer> lstmList;
	public List<TanhLayer> tanhList;
	
	int hiddenLength;
	int linkId;
	
	public SentenceLSTM()
	{
	}
	
	public SentenceLSTM(
			int[] wordIds,
			LookupLayer seedLookup,
			SimplifiedLSTMLayer seedLSTM
		) throws Exception 
	{
		hiddenLength = seedLookup.embeddingLength;
		lookupList = new ArrayList<LookupLayer>();
		
		for(int i = 0; i < wordIds.length; i++)
		{
			LookupLayer tmpLookup = (LookupLayer) seedLookup.cloneWithTiedParams();
			tmpLookup.input[0] = wordIds[i];

			lookupList.add(tmpLookup);
		}
		
		lstmList = new ArrayList<SimplifiedLSTMLayer>();
		tanhList = new ArrayList<TanhLayer>();
		
		for(int i = 0; i < lookupList.size(); i++)
		{
			lstmList.add((SimplifiedLSTMLayer) seedLSTM.cloneWithTiedParams());
			tanhList.add(new TanhLayer(hiddenLength));
		}
		
		// link. important
		for(int i = 0; i < lookupList.size(); i++)
		{
			lookupList.get(i).link(lstmList.get(i), 0);
			if(i > 0)
			{
				tanhList.get(i - 1).link(lstmList.get(i), 1);
			}
			else
			{
			}
			
			lstmList.get(i).link(tanhList.get(i));
		}
		
		linkId = 0;
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		
	}

	@Override
	public void forward() {
		// be careful about the order. it is important.
		for(int i = 0; i < lookupList.size(); i++)
		{
			lookupList.get(i).forward();
			lstmList.get(i).forward();
			tanhList.get(i).forward();
		}
	}

	@Override
	public void backward() {
		// the order is important. Be careful
		for(int i = lookupList.size() - 1; i >= 0 ; i--)
		{
			tanhList.get(i).backward();
			lstmList.get(i).backward();
			lookupList.get(i).backward();
		}
	}

	@Override
	public void update(double learningRate) {
		for(SimplifiedLSTMLayer lstm: lstmList)
		{
			lstm.update(learningRate);
		}
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = lookupList.size() - 1; i >= 0 ; i--)
		{
			tanhList.clear();
			lstmList.clear();
			lookupList.clear();
		}
		
		tanhList.clear();
		lstmList.clear();
		lookupList.clear();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != tanhList.get(tanhList.size() - 1).output.length 
				|| nextIG.length != tanhList.get(tanhList.size() - 1).outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		tanhList.get(tanhList.size() - 1).output = nextI;
		tanhList.get(tanhList.size() - 1).outputG = nextIG;
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
		return tanhList.get(tanhList.size() - 1).output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return tanhList.get(tanhList.size() - 1).outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		return null;
	}
}

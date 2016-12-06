package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import duyuNN.AverageLayer;
import duyuNN.LookupLayer;
import duyuNN.MultiConnectLayer;
import duyuNN.NNInterface;
import duyuNN.SoftmaxLayer;
import duyuNN.TanhLayer;
import duyuNN.combinedLayer.ConnectLinearTanh;
import duyuNN.combinedLayer.ConnectLinearTanhLinear;
import duyuNN.combinedLayer.LookupLinearTanh;
import duyuNN.combinedLayer.SimplifiedLSTMLayer;

public class SentenceLSTMAttention implements NNInterface{
	
	// output is tanhList.get(tanhList.size() - 1)
	List<LookupLayer> lookupList;
	List<SimplifiedLSTMLayer> lstmList;
	List<TanhLayer> tanhList;
	
	List<ConnectLinearTanh> attentionCellList;
	MultiConnectLayer attentionConnect;
	SoftmaxLayer attentionSoftmax;
	
	int hiddenLength;
	int linkId;
	
	double[] targetVec;
	
	public double[] output;
	public double[] outputG;
	
	public SentenceLSTMAttention()
	{
	}
	
	public SentenceLSTMAttention(
			int[] wordIds,
			LookupLayer seedLookup,
			SimplifiedLSTMLayer seedLSTM,
			ConnectLinearTanh seedAttentionCell,
			double[] xTargetVec
		) throws Exception 
	{
		targetVec = new double[xTargetVec.length];
		System.arraycopy(xTargetVec, 0, targetVec, 0, xTargetVec.length);
		
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
		
		if(seedAttentionCell.tanh.output.length != 1)
		{
			throw new Exception("seedAttentionCell.linear2.outputLength != 1");
		}
		
		// add entity attention
		attentionCellList = new ArrayList<ConnectLinearTanh>();
		
		for(int i = 0; i < tanhList.size(); i++)
		{
			attentionCellList.add((ConnectLinearTanh) seedAttentionCell.cloneWithTiedParams());
		}
		
		int[] attentionLengths = new int[attentionCellList.size()];
		Arrays.fill(attentionLengths, 1);
		attentionConnect = new MultiConnectLayer(attentionLengths);
		
		for(int i = 0; i < attentionCellList.size(); i++)
		{
			attentionCellList.get(i).link(attentionConnect, i);
		}
		
		attentionSoftmax = new SoftmaxLayer(attentionConnect.outputLength);
		attentionConnect.link(attentionSoftmax);
		
		output = new double[hiddenLength];
		outputG = new double[hiddenLength];
		
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
		
		// for entity attention
		for(int i = 0; i < tanhList.size(); i++)
		{
			System.arraycopy(targetVec, 0, 
					attentionCellList.get(i).connect.input[0], 0, targetVec.length);
			
			System.arraycopy(tanhList.get(i).output, 0, 
				attentionCellList.get(i).connect.input[1], 0, hiddenLength);
		}
		
		for(int i = 0; i < attentionCellList.size(); i++)
		{
			attentionCellList.get(i).forward();
		}
		
		attentionConnect.forward();
		attentionSoftmax.forward();
		
		for(int i = 0; i < tanhList.size(); i++)
		{
			for(int j = 0; j < hiddenLength; j++)
			{
				output[j] += attentionSoftmax.output[i] * tanhList.get(i).output[j];
			}
		}
	}

	@Override
	public void backward() {
		// we do not backprop for target vector in this code as there is no params there
		for(int i = 0; i < tanhList.size(); i++)
		{
			for(int j = 0; j < hiddenLength; j++)
			{
				attentionSoftmax.outputG[i] += outputG[j] * tanhList.get(i).output[j];
				tanhList.get(i).outputG[j] += outputG[j] * attentionSoftmax.output[i];
			}
		}
		
		attentionSoftmax.backward();
		attentionConnect.backward();
		
		for(int i = 0; i < attentionCellList.size(); i++)
		{
			attentionCellList.get(i).backward();
		}
		
		for(int i = 0; i < attentionCellList.size(); i++)
		{
			for(int j = 0; j < hiddenLength; j++)
			{
				tanhList.get(i).outputG[j] += attentionCellList.get(i).connect.inputG[1][j];
			}
		}
		
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
		
		for(ConnectLinearTanh attentionCell: attentionCellList)
		{
			attentionCell.update(learningRate);
		}
	}
	
	public void update(double learningRate, double attentionLearningRate) {
		for(SimplifiedLSTMLayer lstm: lstmList)
		{
			lstm.update(learningRate);
		}
		
		for(ConnectLinearTanh attentionCell: attentionCellList)
		{
			attentionCell.update(attentionLearningRate);
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
		
		for(ConnectLinearTanh cell: attentionCellList)
		{
			cell.clearGrad();
		}
		
		attentionCellList.clear();
		attentionConnect.clearGrad();
		attentionSoftmax.clearGrad();
		
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

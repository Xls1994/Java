package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import duyuNN.*;
import duyuNN.combinedLayer.*;

// please ignore the Position_1 and Position_2 because their backward functions are not fully implemented
// this backward function is right
// we use an additional position lookup, and learn position embedding with backpropagation.

public class SentenceEntityMemNN_Position_4 implements NNInterface{
	
	List<LookupLayer> lookupList;
	MultiConnectLayer lookupConnect;
	
	int hiddenLength;
	int linkId;
	
	double[] targetVec;
	
	public double[][] outputs;
	public double[][] outputGs;

	NNInterface[] entityTransforms;
	AttentionLayer[] attentionLayers;
	
	List<LookupLayer> positionLookupList;
	List<SigmoidLayer> positionSigmoidList;
	
	int numberOfHops;
	
	public SentenceEntityMemNN_Position_4()
	{
		
	}
	
	public SentenceEntityMemNN_Position_4(
			int[] wordIds,
			LookupLayer seedPositionLookup,
			int[] wordPositions,
			LookupLayer seedLookup,
			NNInterface seedAttentionCell,
 			NNInterface seedEntityTransform,
			double[] xTargetVec,
			int xnumberOfHops
		) throws Exception 
	{
		numberOfHops = xnumberOfHops;
		
		if(wordIds.length != wordPositions.length)
		{
			throw new Exception("wordIds.length != wordPositions.length");
		}
		
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
			
			// do not link here, we calculate this manually at forward.
//			tmpLookup.link(lookupConnect, i);
			
			lookupList.add(tmpLookup);
		}
		
		attentionLayers = new AttentionLayer[numberOfHops];
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i] = new AttentionLayer(seedAttentionCell, xTargetVec.length, wordIds.length);
		}
		
		entityTransforms = new NNInterface[numberOfHops];
		for(int i = 0; i < numberOfHops; i++)
		{
			entityTransforms[i] = (NNInterface) seedEntityTransform.cloneWithTiedParams();
		}
		
		outputs = new double[numberOfHops][];
		outputGs = new double[numberOfHops][];
		
		for(int i = 0; i < numberOfHops; i++)
		{
			outputs[i] = new double[hiddenLength];
			outputGs[i] = new double[hiddenLength];
		}
		
		positionLookupList = new ArrayList<LookupLayer>();
		positionSigmoidList = new ArrayList<SigmoidLayer>();
		
		for(int i = 0; i < wordPositions.length; i++)
		{
			LookupLayer tmpLookup = (LookupLayer) seedPositionLookup.cloneWithTiedParams();
			tmpLookup.input[0] = wordPositions[i];
			
			SigmoidLayer tmpSigmoid = new SigmoidLayer(seedPositionLookup.embeddingLength); 
			tmpLookup.link(tmpSigmoid);
			
			positionLookupList.add(tmpLookup);
			positionSigmoidList.add(tmpSigmoid);
		}
		
		linkId = 0;
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		
	}

	@Override
	public void forward() {
		
		for(int i = 0; i < positionLookupList.size(); i++)
		{
			positionLookupList.get(i).forward();
			positionSigmoidList.get(i).forward();
		}
		
		for(int i = 0; i < lookupList.size(); i++)
		{
			lookupList.get(i).forward();
		}
		
		// be careful
		for(int i = 0; i < lookupList.size(); i++)
		{
			for(int j = 0; j < lookupList.get(i).output.length; j++)
				// we regard lookup-sigmoid as a neural gate
				lookupConnect.input[i][j] = lookupList.get(i).output[j] * positionSigmoidList.get(i).output[j]; 
		}
		
		lookupConnect.forward();
		
		// i should start from 0 and end until numberOfHops - 1
		for(int i = 0; i < numberOfHops; i++)
		{
			if(i == 0)
			{
				attentionLayers[i].forward(targetVec, lookupConnect.output);
				System.arraycopy(targetVec, 0, 
						(double[]) entityTransforms[i].getInput(0), 0, targetVec.length);
			}
			else
			{
				attentionLayers[i].forward(outputs[i-1], lookupConnect.output);
				System.arraycopy(outputs[i-1], 0, 
						(double[]) entityTransforms[i].getInput(0), 0, outputs[i-1].length);
			}
			
			entityTransforms[i].forward();
			
			for(int j = 0; j < outputs[i].length; j++)
			{
				outputs[i][j] = ((double[]) entityTransforms[i].getOutput(0))[j] 
						+ attentionLayers[i].output[j];
			}
		}
	}

	@Override
	public void backward() {
		
		// i should start from numberOfHops - 1 and end until 0
		for(int i = numberOfHops - 1; i >= 0 ; i--)
		{
			for(int j = 0; j < outputs[i].length; j++)
			{
				attentionLayers[i].outputG[j] += outputGs[i][j];
				((double[]) entityTransforms[i].getOutputG(0) )[j] += outputGs[i][j];
			}
			
			entityTransforms[i].backward();
			attentionLayers[i].backward();
			
			for(int j = 0; j < attentionLayers[i].contextGs.length; j++)
			{
				lookupConnect.outputG[j] += attentionLayers[i].contextGs[j];
			}
			
			if(i > 0)
			{
				for(int j = 0; j < outputGs[i-1].length; j++)
				{
					outputGs[i-1][j] += attentionLayers[i].inputG[j]
							+ ( (double[]) entityTransforms[i].getInputG(0) )[j];
				}
			}
		}
		
		lookupConnect.backward();
		
		// be careful about the positionLookup
		for(int i = 0; i < positionSigmoidList.size(); i++)
		{
			for(int j = 0; j < positionSigmoidList.get(i).output.length; j++)
				positionSigmoidList.get(i).outputG[j] += lookupConnect.inputG[i][j] * lookupList.get(i).output[j];
		}
		
		for(int i = 0; i < positionSigmoidList.size(); i++)
		{
			positionSigmoidList.get(i).backward();
			positionLookupList.get(i).backward();
		}
	}

	@Override
	public void update(double learningRate) {
		
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i].update(learningRate);
			entityTransforms[i].update(learningRate);
		}
		for(int i = 0; i < positionLookupList.size(); i++)
		{
			positionLookupList.get(i).update(learningRate);
		}
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i].updateAdaGrad(learningRate, 1);
			entityTransforms[i].updateAdaGrad(learningRate, 1);
		}
		for(int i = 0; i < positionLookupList.size(); i++)
		{
			positionLookupList.get(i).updateAdaGrad(learningRate, 1);
		}
	}

	@Override
	public void clearGrad() {
		
		for(int i = lookupList.size() - 1; i >= 0 ; i--)
		{
			lookupList.get(i).clearGrad();
		}
		lookupList.clear();
		
		lookupConnect.clearGrad();
		
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i].clearGrad();
			entityTransforms[i].clearGrad();
			
			Arrays.fill(outputs[i], 0);
			Arrays.fill(outputGs[i], 0);
		}
		
		for(LookupLayer positionLookup: positionLookupList)
		{
			positionLookup.clearGrad();
		}
		positionLookupList.clear();
		
		for(SigmoidLayer layer: positionSigmoidList)
		{
			layer.clearGrad();
		}
		positionSigmoidList.clear();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != outputs[numberOfHops - 1].length 
				|| nextIG.length != outputGs[numberOfHops - 1].length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		outputs[numberOfHops - 1] = nextI;
		outputGs[numberOfHops - 1] = nextIG;
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
		return outputs[numberOfHops - 1];
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputGs[numberOfHops - 1];
	}

	@Override
	public Object cloneWithTiedParams() {
		return null;
	}
}

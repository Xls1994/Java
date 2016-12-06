package duyuNN.combinedLayer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.Random;

import duyuNN.LinearLayer;
import duyuNN.MultiConnectLayer;
import duyuNN.NNInterface;
import duyuNN.SigmoidLayer;
import duyuNN.TanhLayer;

public class HighWayNN implements NNInterface{

	// current input linkId = 0
	// history linkId = 1. This is important!!!!! See the last line of forward function.
	
	// A simplification: let h_(t-1) = c_(t-1)
	public double[] output;
	public double[] outputG;
	
	// connect input and previous output
	public MultiConnectLayer connectInputHistory;
	
	// input gate
	LinearLayer inputLinear;
	SigmoidLayer inputSigmoid;
	
	// candidate memory cell
	LinearLayer candidateStateLinear;
	TanhLayer candidateStateTanh;
	
	int hiddenLength;
	
	public HighWayNN() {
		
	}
	
	public void dumpToStream(Writer bw) throws IOException
    {
		bw.write(hiddenLength + "\n");
		inputLinear.dumpToStream(bw);
		candidateStateLinear.dumpToStream(bw);
    }
    
    public static SimplifiedLSTMLayer loadFromStream(BufferedReader br) 
    		throws Exception 
    {
    	int _hiddenLength = Integer.parseInt(br.readLine());
    	
    	LinearLayer _inputLinear = LinearLayer.loadFromStream(br);
    	LinearLayer _forgetLinear = LinearLayer.loadFromStream(br);
    	LinearLayer _candidateStateLinear = LinearLayer.loadFromStream(br);
    	SimplifiedLSTMLayer layer = new SimplifiedLSTMLayer(_inputLinear, _forgetLinear, _candidateStateLinear, _hiddenLength);
    	
    	return layer;
    }
	
	public HighWayNN(int xHiddenLength) throws Exception
	{
		hiddenLength = xHiddenLength;
		
		connectInputHistory = new MultiConnectLayer(new int[]{hiddenLength, hiddenLength});
		// connectInputPreOutput will link to inputLinear, forgetLinear and candidateStateLinear
		// I did not link it to any of these three layers. 
		// I manually link them in forward and backward.
	
		inputLinear = new LinearLayer(2 * hiddenLength, hiddenLength);
		inputSigmoid = new SigmoidLayer(hiddenLength);
		inputLinear.link(inputSigmoid);		
		
		candidateStateLinear = new LinearLayer(2 * hiddenLength, hiddenLength);
		candidateStateTanh = new TanhLayer(hiddenLength);
		candidateStateLinear.link(candidateStateTanh);
		
		output = new double[hiddenLength];
		outputG = new double[hiddenLength];
		
	}
	
	public HighWayNN(
			LinearLayer xseedInputLinear,
			LinearLayer xseedCandidateStatelinear,
			int xHiddenLength) throws Exception
	{
		hiddenLength = xHiddenLength;
		
		if(	!(hiddenLength == xseedInputLinear.inputLength/2 &&
				hiddenLength == xseedInputLinear.outputLength &&
				hiddenLength == xseedCandidateStatelinear.inputLength/2 &&
				hiddenLength == xseedCandidateStatelinear.outputLength))
		{
			System.err.println("WRONG!!!! lengthes do not match");
		}
		
		connectInputHistory = new MultiConnectLayer(new int[]{hiddenLength, hiddenLength});
		// connectInputPreOutput will link to inputLinear, forgetLinear and candidateStateLinear
		// I did not link it to any of these three layers. 
		// I manually link them in forward and backward.
	
		inputLinear = (LinearLayer) xseedInputLinear.cloneWithTiedParams();
		inputSigmoid = new SigmoidLayer(hiddenLength);
		inputLinear.link(inputSigmoid);		
		
		candidateStateLinear = (LinearLayer) xseedCandidateStatelinear.cloneWithTiedParams();
		candidateStateTanh = new TanhLayer(hiddenLength);
		candidateStateLinear.link(candidateStateTanh);
		
		output = new double[hiddenLength];
		outputG = new double[hiddenLength];
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		inputLinear.randomize(r, min, max);
		candidateStateLinear.randomize(r, min, max);
	}

	@Override
	public void forward() {
		
		connectInputHistory.forward();
		
		// link manually
		System.arraycopy(connectInputHistory.output, 0, 
				inputLinear.input, 0, hiddenLength * 2);
		System.arraycopy(connectInputHistory.output, 0, 
				candidateStateLinear.input, 0, hiddenLength * 2);
		
		inputLinear.forward();
		inputSigmoid.forward();
		
		candidateStateLinear.forward();
		candidateStateTanh.forward();
		
		for(int i = 0; i < hiddenLength; i++)
		{
			output[i] = inputSigmoid.output[i] * connectInputHistory.input[0][i] +
					(1 - inputSigmoid.output[i]) * candidateStateTanh.output[i];
		}
	}

	@Override
	public void backward() {
		for(int i = 0; i < hiddenLength; i++)
		{
			inputSigmoid.outputG[i] = outputG[i] * (connectInputHistory.input[0][i] - candidateStateTanh.output[i]);
			candidateStateTanh.outputG[i] = outputG[i] * (1 - inputSigmoid.output[i]);
		}
		
		inputSigmoid.backward();
		inputLinear.backward();
		
		candidateStateTanh.backward();
		candidateStateLinear.backward();
		
		for(int i = 0; i < 2 * hiddenLength; i++)
		{
			connectInputHistory.outputG[i] = inputLinear.inputG[i] +
								 candidateStateLinear.inputG[i];
		}
		connectInputHistory.backward();
		
		// don't forget this step.
		for(int i = 0; i < hiddenLength; i++)
		{
			connectInputHistory.inputG[0][i] += outputG[i] * inputSigmoid.output[i];
		}
	}

	@Override
	public void update(double learningRate) {
		inputLinear.update(learningRate);
		candidateStateLinear.update(learningRate);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	@Override
	public void clearGrad() {
		connectInputHistory.clearGrad();
		
		inputLinear.clearGrad();
		inputSigmoid.clearGrad();
		
		candidateStateLinear.clearGrad();
		candidateStateTanh.clearGrad();
		
		Arrays.fill(outputG, 0);
		Arrays.fill(output, 0);
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[]) nextInput;
		double[] nextIG = (double[]) nextInputG; 
		
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
		link(nextLayer, 0);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return connectInputHistory.input[id];
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connectInputHistory.inputG[id];
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		
		HighWayNN clone = null;
		try {
			clone = new HighWayNN(inputLinear,
					candidateStateLinear,
					hiddenLength);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return clone;
	}
}

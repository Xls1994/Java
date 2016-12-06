package duyuNN.combinedLayer;
//package duyuNN.combinedLayer;
//
//import java.util.Arrays;
//import java.util.Random;
//
//import duyuNN.ElementMultiLayer;
//import duyuNN.LinearLayer;
//import duyuNN.MultiConnectLayer;
//import duyuNN.NNInterface;
//import duyuNN.SigmoidLayer;
//import duyuNN.TanhLayer;
//
//public class AnotherSimplifiedLSTMLayer implements NNInterface{
//
//	// current input linkId = 0
//	// history linkId = 1. This is important!!!!! See the last line of forward function.
//	
//	// A simplification: let h_(t-1) = c_(t-1)
//	public double[] output;
//	public double[] outputG;
//	
//	// input gate
//	LinearLayer inputLinear;
//	SigmoidLayer inputSigmoid;
//	
//	// forget gate
//	LinearLayer forgetLinear;
//	SigmoidLayer forgetSigmoid;
//	
//	// candidate memory cell
//	LinearLayer candidateStateLinear;
//	TanhLayer candidateStateTanh;
//	
//	int outputLength;
//	int inputLength;
//	
//	double[] input;
//	double[] inputG;
//	
//	public AnotherSimplifiedLSTMLayer() {
//		
//	}
//	
//	public AnotherSimplifiedLSTMLayer(
//			int xInputLength,
//			int xOutputLength) throws Exception
//	{
//		outputLength = xOutputLength;
//		inputLength = xInputLength;
//		
//		// input is actually the concatenation of current input vec and previous history.
//		input  = new double[inputLength + outputLength];
//		inputG = new double[inputLength + outputLength];
//		
//		// I did not link it to any of these three layers. 
//		// I manually link them in forward and backward.
//		
//		inputLinear = new LinearLayer(inputLength + outputLength, outputLength);
//		inputSigmoid = new SigmoidLayer(outputLength);
//		inputLinear.link(inputSigmoid);		
//		
//		forgetLinear = new LinearLayer(inputLength + outputLength, outputLength);
//		forgetSigmoid = new SigmoidLayer(outputLength);
//		forgetLinear.link(forgetSigmoid);
//		
//		candidateStateLinear = new LinearLayer(inputLength + outputLength, outputLength);
//		candidateStateTanh = new TanhLayer(outputLength);
//		candidateStateLinear.link(candidateStateTanh);
//		
//		output = new double[outputLength];
//		outputG = new double[outputLength];
//	}
//	
//	public AnotherSimplifiedLSTMLayer(int xHiddenLength) throws Exception
//	{
//		this(xHiddenLength, xHiddenLength);
//	}
//	
//	public AnotherSimplifiedLSTMLayer(
//			LinearLayer xseedInputLinear,
//			LinearLayer xseedForgetLinear,
//			LinearLayer xseedCandidateStatelinear,
//			int xInputLength,
//			int xHiddenLength) throws Exception
//	{
//		outputLength = xHiddenLength;
//		inputLength = xInputLength;
//		
//		if(	!(  inputLength + outputLength == xseedInputLinear.inputLength &&
//				outputLength == xseedInputLinear.outputLength &&
//				inputLength + outputLength == xseedForgetLinear.inputLength &&
//				outputLength == xseedForgetLinear.outputLength &&
//				inputLength + outputLength == xseedCandidateStatelinear.inputLength &&
//				outputLength == xseedCandidateStatelinear.outputLength))
//		{
//			System.err.println("WRONG!!!! lengthes do not match");
//		}
//		
//		inputLinear = (LinearLayer) xseedInputLinear.cloneWithTiedParams();
//		inputSigmoid = new SigmoidLayer(outputLength);
//		inputLinear.link(inputSigmoid);		
//		
//		forgetLinear = (LinearLayer) xseedForgetLinear.cloneWithTiedParams();
//		forgetSigmoid = new SigmoidLayer(outputLength);
//		forgetLinear.link(forgetSigmoid);
//		
//		candidateStateLinear = (LinearLayer) xseedCandidateStatelinear.cloneWithTiedParams();
//		candidateStateTanh = new TanhLayer(outputLength);
//		candidateStateLinear.link(candidateStateTanh);
//		
//		output = new double[outputLength];
//		outputG = new double[outputLength];
//	}
//	
//	@Override
//	public void randomize(Random r, double min, double max) {
//		inputLinear.randomize(r, min, max);
//		forgetLinear.randomize(r, min, max);
//		candidateStateLinear.randomize(r, min, max);
//	}
//
//	@Override
//	public void forward() {
//
//		// link manually
//		System.arraycopy(input, 0, 
//				inputLinear.input, 0, inputLength + outputLength);
//		System.arraycopy(input, 0, 
//				forgetLinear.input, 0, inputLength + outputLength);
//		System.arraycopy(input, 0, 
//				candidateStateLinear.input, 0, inputLength + outputLength);
//		
//		inputLinear.forward();
//		inputSigmoid.forward();
//		
//		forgetLinear.forward();
//		forgetSigmoid.forward();
//		
//		candidateStateLinear.forward();
//		candidateStateTanh.forward();
//		
//		for(int i = 0; i < outputLength; i++)
//		{
//			// connectInputHistory.input[1] is the previous history.
//			output[i] = inputSigmoid.output[i] *  candidateStateTanh.output[i] +
//					forgetSigmoid.output[i] * input[inputLength + i];
//		}
//	}
//
//	@Override
//	public void backward() {
//		for(int i = 0; i < outputLength; i++)
//		{
//			inputSigmoid.outputG[i] = outputG[i] * candidateStateTanh.output[i];
//			candidateStateTanh.outputG[i] = outputG[i] * inputSigmoid.output[i];
//			
//			forgetSigmoid.outputG[i] = outputG[i] * connectInputHistory.input[1][i];
//			// don't forget to add to connectInputPreOutput.inputG[1][i] at the end.
//		}
//		
//		inputSigmoid.backward();
//		inputLinear.backward();
//		
//		forgetSigmoid.backward();
//		forgetLinear.backward();
//		
//		candidateStateTanh.backward();
//		candidateStateLinear.backward();
//		
//		for(int i = 0; i < inputLength + outputLength; i++)
//		{
//			connectInputHistory.outputG[i] = inputLinear.inputG[i] +
//								forgetLinear.inputG[i] + candidateStateLinear.inputG[i];
//		}
//		connectInputHistory.backward();
//		
//		// don't forget this step.
//		for(int i = 0; i < outputLength; i++)
//		{
//			connectInputHistory.inputG[1][i] += outputG[i] * forgetSigmoid.output[i];
//		}
//	}
//
//	@Override
//	public void update(double learningRate) {
//		inputLinear.update(learningRate);
//		forgetLinear.update(learningRate);
//		candidateStateLinear.update(learningRate);
//	}
//
//	@Override
//	public void updateAdaGrad(double learningRate, int batchsize) {
//		// TODO Auto-generated method stub
//		
//	}
//
//	@Override
//	public void clearGrad() {
//		connectInputHistory.clearGrad();
//		
//		inputLinear.clearGrad();
//		inputSigmoid.clearGrad();
//		
//		forgetLinear.clearGrad();
//		forgetSigmoid.clearGrad();
//		
//		candidateStateLinear.clearGrad();
//		candidateStateTanh.clearGrad();
//		
//		Arrays.fill(outputG, 0);
//		Arrays.fill(output, 0);
//	}
//
//	@Override
//	public void link(NNInterface nextLayer, int id) throws Exception {
//		Object nextInputG = nextLayer.getInputG(id);
//		Object nextInput = nextLayer.getInput(id);
//		
//		double[] nextI = (double[]) nextInput;
//		double[] nextIG = (double[]) nextInputG; 
//		
//		if(nextI.length != output.length || nextIG.length != outputG.length)
//		{
//			throw new Exception("The Lengths of linked layers do not match.");
//		}
//		
//		output = nextI;
//		outputG = nextIG;
//	}
//
//	@Override
//	public void link(NNInterface nextLayer) throws Exception {
//		// TODO Auto-generated method stub
//		link(nextLayer, 0);
//	}
//
//	@Override
//	public Object getInput(int id) {
//		// TODO Auto-generated method stub
//		return connectInputHistory.input[id];
//	}
//
//	@Override
//	public Object getOutput(int id) {
//		// TODO Auto-generated method stub
//		return output;
//	}
//
//	@Override
//	public Object getInputG(int id) {
//		// TODO Auto-generated method stub
//		return connectInputHistory.inputG[id];
//	}
//
//	@Override
//	public Object getOutputG(int id) {
//		// TODO Auto-generated method stub
//		return outputG;
//	}
//
//	@Override
//	public Object cloneWithTiedParams() {
//		
//		AnotherSimplifiedLSTMLayer clone = null;
//		try {
//			clone = new AnotherSimplifiedLSTMLayer(
//					inputLinear,
//					forgetLinear,
//					candidateStateLinear,
//					inputLength,
//					outputLength);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		
//		return clone;
//	}
//}

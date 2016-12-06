package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.*;

public class ConnectLinearTanhLinear implements NNInterface {

	public MultiConnectLayer connect;
	public LinearLayer linear1;
	public TanhLayer tanh;
	public LinearLayer linear2;
	
	public ConnectLinearTanhLinear() {
		// TODO Auto-generated constructor stub
	}
	
	public ConnectLinearTanhLinear(int xConnectInputLength1,
			int xConnectInputLength2,
			int hiddenLength, 
			int outputLength) throws Exception
	{
		connect = new MultiConnectLayer(new int[]{xConnectInputLength1, xConnectInputLength2});
		linear1 = new LinearLayer(connect.outputLength, hiddenLength);
		tanh = new TanhLayer(hiddenLength);
		linear2 = new LinearLayer(hiddenLength, outputLength);
		
		connect.link(linear1);
		linear1.link(tanh);
		tanh.link(linear2);
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear1.randomize(r, min, max);
		linear2.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		connect.forward();
		linear1.forward();
		tanh.forward();
		linear2.forward();
	}

	@Override
	public void backward() {
		linear2.backward();
		tanh.backward();
		linear1.backward();
		connect.backward();
	}
	
	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		linear1.update(learningRate);
		linear2.update(learningRate);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		connect.clearGrad();
		linear1.clearGrad();
		tanh.clearGrad();
		linear2.clearGrad();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
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

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, 0);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return connect.getInput(id);
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return linear2.getOutput(id);
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connect.getInputG(id);
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return linear2.getOutputG(id);
	}

	@Override
	public Object cloneWithTiedParams() {
		ConnectLinearTanhLinear clone = new ConnectLinearTanhLinear();
		
		clone.connect = (MultiConnectLayer) connect.cloneWithTiedParams();
		clone.linear1 = (LinearLayer) linear1.cloneWithTiedParams();
		clone.tanh = (TanhLayer) tanh.cloneWithTiedParams();
		clone.linear2 = (LinearLayer) linear2.cloneWithTiedParams();
		
		try {
			clone.connect.link(clone.linear1);
			clone.linear1.link(clone.tanh);
			clone.tanh.link(clone.linear2);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

}

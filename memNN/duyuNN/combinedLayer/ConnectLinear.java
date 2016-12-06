package duyuNN.combinedLayer;

import java.util.Random;

import duyuNN.*;

public class ConnectLinear implements NNInterface {

	public MultiConnectLayer connect;
	public LinearLayer linear;
	
	public ConnectLinear() {
		// TODO Auto-generated constructor stub
	}
	
	public ConnectLinear(int xConnectInputLength1,
			int xConnectInputLength2,
			int hiddenLength) throws Exception
	{
		connect = new MultiConnectLayer(new int[]{xConnectInputLength1, xConnectInputLength2});
		linear = new LinearLayer(connect.outputLength, hiddenLength);
		
		connect.link(linear);
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		connect.forward();
		linear.forward();
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		linear.backward();
		connect.backward();
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		linear.update(learningRate);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		connect.clearGrad();
		linear.clearGrad();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != linear.output.length 
				|| nextIG.length != linear.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		linear.output = nextI;
		linear.outputG = nextIG;
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
		return linear.output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connect.getInputG(id);
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return linear.outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		ConnectLinear clone = new ConnectLinear();
		
		clone.connect = (MultiConnectLayer) connect.cloneWithTiedParams();
		clone.linear = (LinearLayer) linear.cloneWithTiedParams();
		
		try {
			clone.connect.link(clone.linear);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

}

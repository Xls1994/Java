package dataprepare;

public class Data {

	public int goldPol;
	public int predictedPol;
	
	public String text;
	public String target;
	
	public Data(String xText,
			int goldRating,
			String xTarget) {
		
		this.goldPol = goldRating;
		this.predictedPol = -1;
		this.text = xText;
		this.target = xTarget;
	}
}

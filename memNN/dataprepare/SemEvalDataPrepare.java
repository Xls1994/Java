package dataprepare;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;

import org.dom4j.Document;
import org.dom4j.Element;

public class SemEvalDataPrepare {

	static StanfordWordSegMain stanfordMain = null;
	
	static void run(String file) throws IOException
	{
		PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
		          new FileOutputStream(file + ".norm.addConf"), "utf8")));
		PrintWriter writerSeg = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
		          new FileOutputStream(file + ".seg.addConf"), "utf8")));
		
		Document doc = DomUtil.parseDom(new File(file) );

		Element root = doc.getRootElement();

		List<Element> sentences = root.elements();
		System.out.println("sentences.size(): " + sentences.size());
		
		for(Element sentence: sentences)
		{
			String id = sentence.attributeValue("id");
			
			System.out.println("id " + id);
			
			Element textElement = sentence.element("text");
			Element tmpTerms = sentence.element("aspectTerms");
			
			if(tmpTerms == null)
			{
				System.err.println(id + "\ttmpTerms == null");
				continue;
			}
            List<Element> terms = tmpTerms.elements();
            
            String textSrc = textElement.getText();
            for(Element term: terms)
            {
            	String aspectTerm = term.attributeValue("term");
            	String polarity = term.attributeValue("polarity");
            	
            	int startId = Integer.parseInt(term.attributeValue("from"));
            	int endId = Integer.parseInt(term.attributeValue("to"));
            	
            	String textCopy = textSrc;
            	
            	System.out.println("text:" + textCopy);
            	System.out.println("aspectTerm: " + aspectTerm);
            	System.out.println("polarity: " + polarity);
            	System.out.println("start-end ids: " + startId + "-" + endId);
            	
            	String target = textCopy.substring(startId, endId);
            	if(!target.equals(aspectTerm))
            	{
            		System.out.println("target != aspectTerm");
            		System.out.println("target: " + target);
            		System.out.println("aspectTerm: " + aspectTerm);
            	}
            	
            	String modifiedText = "";
            	
            	if(startId > 0)
            		modifiedText += textCopy.substring(0, startId);
            	
            	modifiedText += "$T$";
            	
            	if(endId != textCopy.length())
            		modifiedText += textCopy.substring(endId);
            	
            	System.out.println("modifiedText: " + modifiedText);

            	List<String> results = stanfordMain.run(modifiedText, false);
            	String segText = "";
            	for(int i = 0; i < results.size(); i++)
            	{
            		String seg = results.get(i);
            		segText = segText + " " + seg;
            	}
            	segText = segText.replace("$ T$", "$T$");
            	segText = segText.replace("do n't", "don't");
            	
            	System.out.println("segText: " + segText);
            	results.clear();
            	
//            	if(polarity.equals("positive") || polarity.equals("negative") || polarity.equals("neutral"))
//            	{
        		writer.write(modifiedText + "\n");
        		writer.write(target + "\n");
        		
        		writerSeg.write(segText.trim() + "\n");
        		writerSeg.write(target + "\n");
        		
        		if(polarity.equals("positive"))
        		{
        			writer.write("1\n");
        			writerSeg.write("1\n");
        		}
        		else if(polarity.equals("negative"))
        		{
        			writer.write("2\n");
        			writerSeg.write("2\n");
        		}
        		else if(polarity.equals("neutral"))
        		{
        			writer.write("0\n");
        			writerSeg.write("0\n");
        		}
        		else if(polarity.equals("conflict"))
        		{
        			writer.write("3\n");
        			writerSeg.write("3\n");
        		}
        		else
        		{
        			System.err.println("wrong label format!!!");
        		}
//            	}
//            	else
//            	{
//            		writer.write(modifiedText + "\n");
//            		writer.write(target + "\n");
//            		writer.write("2\n");
//            	}
            	
            }
		}
		
		doc.clearContent();
		
		writer.close();
		writerSeg.close();
	}
	
	public static void main(String[] args) {
		
		stanfordMain = new StanfordWordSegMain();
		
		try {
			run("corpus\\Restaurants_Train.xml");
//			run("/home/laboratory/memoryCorpus/restaurants_trial.xml");
//			run("/home/laboratory/memoryCorpus/laptops_train.xml");
//			run("/home/laboratory/memoryCorpus/laptops_trial.xml");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}

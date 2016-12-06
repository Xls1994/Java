package dataprepare;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class StanfordWordSegMain {
	StanfordCoreNLP pipeline = null;
	
	public StanfordWordSegMain()
	{
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
	    Properties props = new Properties();
	    props.put("annotators", "tokenize, ssplit");
	    pipeline = new StanfordCoreNLP(props);
	}
	
	public List<String> run(String text, boolean isSaveSentenceSplitor)
	{
		Annotation document = new Annotation(text);
	    pipeline.annotate(document);

	    List<String> results = new ArrayList<String>();
	    
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    for(int i = 0; i < sentences.size(); i++) {
	    	CoreMap sentence = sentences.get(i);
	    	for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
	    		String word = token.get(TextAnnotation.class);
	    		
	    		results.add(word);
	    	}
	    	
	    	if(isSaveSentenceSplitor && i < sentences.size() - 1)
	    	{
	    		results.add("<sssss>"); 
	    	}
	    }
	    
	    return results;
	}
	
	public static void main(String[] args)
	{
		StanfordWordSegMain main = new StanfordWordSegMain();
		
		String text = "Stanford University is located in California. It is a great university, founded in 1891.";// Add your text here!
		text = "To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.";
		text = "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.";
		List<String> results = main.run(text, true);
		for(String result: results)
		{
			System.out.println(result);
		}
	}
}

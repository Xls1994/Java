package dataprepare;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

/*
 * This class is to call StanfordCore for conducting lemmatizationg, tagging and noun extraction.
 * You need to download the nlpcore at http://nlp.stanford.edu/software/corenlp.shtml
 * 
 * Afterwards, you should add the ejml-0.23.jar, joda-time.jar, jollyday.jar,
 * stanford-corenlp-3.5.0-model.jar, stanf0rd-corenlp-3.5.0.jar, xom.jar 
 * to the build path of this project
 * 
 */

class ParseResult{
	public String word;
	public String postag;
	
	public ParseResult(String word, String postag) {
		this.word = word;
		this.postag = postag;
	}
}

public class StanfordParserMain {

	StanfordCoreNLP pipeline = null;
	
	public StanfordParserMain()
	{
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
	    Properties props = new Properties();
	    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse");
	    pipeline = new StanfordCoreNLP(props);
	}
	
	public List<ParseResult> run(String text)
	{
		Annotation document = new Annotation(text);
	    pipeline.annotate(document);

	    List<ParseResult> results = new ArrayList<ParseResult>();
	    
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    for(CoreMap sentence: sentences) {
	    	for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
	    		String word = token.get(TextAnnotation.class);
	    		String pos = token.get(PartOfSpeechAnnotation.class);
	    		
	    		results.add(new ParseResult(word, pos));
	    	}
	    	Tree tree = sentence.get(TreeAnnotation.class);
	    	System.out.println(tree.toString());
	    }
	    
	    return results;
	}
	
	private void printResutls(List<ParseResult> results)
	{
		for(int i = 0; i < results.size(); i++)
		{
			System.out.println(	results.get(i).word + " / " + 
								results.get(i).postag);
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String text = "Stanford University is located in California. It is a great university, founded in 1891.";// Add your text here!
		
	    StanfordParserMain main = new StanfordParserMain();
	    List<ParseResult> results = main.run(text);
	    main.printResutls(results);
	}
}

package dataprepare;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Funcs {
	
//	public static DecimalFormat dFormat = new DecimalFormat("0.0000000000");

	public static double[][] loadEmbeddingFile(
			String embedFile,
			int embeddingLength,
			String encoding,
			boolean isL2Norm,
			HashMap<String, Integer> vocabMap, 
			HashSet<String> wordSet) throws IOException
	{
		TreeMap<String, double[]> tmpTable = new TreeMap<String, double[]>();
		
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(embedFile) , encoding));
		
		String line = null;
		while((line = reader.readLine()) != null)
		{
			String[] splits = line.split(" |\t");
			
			String word = splits[0];
			if(!wordSet.contains(word))
			{
				continue;
			}
			double[] values = new double[embeddingLength];
			
			for(int j = 1; j < splits.length; j++)
			{
				Double value = Double.parseDouble(splits[j]);
				values[j-1] = value;
			}
			
			if(isL2Norm)
			{
				l2Norm(values);
			}
			tmpTable.put(word, values);
		}
		reader.close();
		
		// for output
		int idx = 0;
		
		double[][] table = new double[tmpTable.size()][];
		for(String key: tmpTable.keySet())
		{
			table[idx] = new double[embeddingLength];
			vocabMap.put(key, idx);
			
			System.arraycopy(tmpTable.get(key), 0, table[idx], 0, embeddingLength);
			idx++;
		}
		System.out.println("vocabMap.size(): " + vocabMap.size());
		
		return table;
	}
	
//	public static void loadEmbeddingFile(String embedFile,
//			int embeddingLength,
//			String encoding,
//			boolean isL2Norm,
//			HashMap<String, Integer> vocabMap,
//			double[][] table) throws IOException
//	{
//		BufferedReader reader = new BufferedReader(new InputStreamReader(
//				new FileInputStream(embedFile) , encoding));
//		String line = null;
//		
//		int idx = 0;
//		while((line = reader.readLine()) != null)
//		{
//			String[] splits = line.split(" |\t");
//			
//			String word = splits[0];
//			vocabMap.put(word, idx);
//			table[idx] = new double[embeddingLength];
//			
//			for(int j = 1; j < splits.length; j++)
//			{
//				Double value = Double.parseDouble(splits[j]);
//				table[idx][j-1] = value;
//			}
//			// to be debuged
//			if(isL2Norm)
//			{
//				l2Norm(table[idx]);
//			}
//			
//			idx++;
//		}
//		reader.close();
//	}
//	
//	
//	public static HashMap<String, Double[]> loadEmbedFile(String embedFile,
//			int embeddingLength,
//			String encoding,
//			boolean isL2Norm) throws IOException
//	{
//		HashMap<String, Double[]> lookupTable = new HashMap<String, Double[]>();
//		
//		BufferedReader reader = new BufferedReader(new InputStreamReader(
//				new FileInputStream(embedFile) , encoding));
//		String line = null;
//		
//		while((line = reader.readLine()) != null)
//		{
//			String[] splits = line.split(" |\t");
//			
//			if(splits.length != embeddingLength + 1)
//				continue;
//			
//			String word = splits[0];
//			Double[] embedValues = new Double[embeddingLength];
//			
//			for(int j = 1; j < splits.length; j++)
//			{
//				Double value = Double.parseDouble(splits[j]);
//				embedValues[j-1] = value;
//			}
//			
//			if(isL2Norm)
//			{
//				l2Norm(embedValues);
//			}
//			
//			lookupTable.put(word, embedValues);
//		}
//		reader.close();
//		
//		return lookupTable;
//	}
	
	public static void l1Norm(Double[] values)
	{
		Double Z = 0.0;
		for(int i = 0; i < values.length; i++)
		{
			Z = Z + Math.abs(values[i]);
		}
		
		for(int i = 0; i < values.length; i++)
		{
			values[i] = values[i] * 1.0 / Z;
		}
	}
	
	public static void l2Norm(Double[] values)
	{
		Double Z = 0.0;
		for(int i = 0; i < values.length; i++)
		{
			Z = Z + values[i] * values[i];
		}
		
		for(int i = 0; i < values.length; i++)
		{
			values[i] = values[i] * 1.0 / Z;
		}
	}
	
	public static void l2Norm(double[] values)
	{
		Double Z = 0.0;
		for(int i = 0; i < values.length; i++)
		{
			Z = Z + values[i] * values[i];
		}
		
		for(int i = 0; i < values.length; i++)
		{
			values[i] = values[i] * 1.0 / Z;
		}
	}
	
	public static HashMap<String, String> parseArgs(String[] args)
	{
		HashMap<String, String> argMap = new HashMap<String, String>();
		
		for(int i = 0; i < args.length; i++)
		{
			String key = args[i];
			if(key.startsWith("-"))
			{
				if(i + 1 < args.length)
				{
					argMap.put(args[i], args[i + 1]);
					i++;
				}
			}
		}
		return argMap;
	}
	
	public static List<Data> loadCorpus(
			String inPath, 
			String encoding)
	{
		List<Data> dataList = new ArrayList<Data>();
		List<String> sourceList = IO.readFile(inPath, "utf8");
		
		for(int i = 0; i < sourceList.size(); i+=3)
		{
			String srcText = sourceList.get(i).toLowerCase();
			String targetStr = sourceList.get(i + 1).toLowerCase();
			int goldPol = Integer.parseInt(sourceList.get(i + 2));
			
			if(goldPol == -1)
				goldPol = 2;
			
			dataList.add(new Data(srcText, goldPol, targetStr));
		}
		
		return dataList;
	}
	
	public static void loadVocabFromFile(String vocabFile, 
			HashMap<String, Integer> vocabMap,
			String encoding)
	{
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(vocabFile) , encoding));
			String line = null;
			while((line = reader.readLine()) != null)
			{
				String[] words = line.split(" |\t");
				if(words.length < 2)
				{
					System.out.println(line);
				}
				String word = words[0];
				int idx = Integer.parseInt(words[1]);
				
				vocabMap.put(word, idx);
			}
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static void filterMapWithFreq(
			HashMap<String, Integer> origFreatureFreqMap,
			int minFreq,
			HashMap<String, Integer> vocabMap)
	{
		TreeMap<Integer, List<String>> treeMap = new TreeMap<Integer, List<String>>();
		
		for(String word: origFreatureFreqMap.keySet())
		{
			int freq = origFreatureFreqMap.get(word);
			if(freq >= minFreq)
			{
				if(!treeMap.containsKey(freq))
				{
					treeMap.put(freq, new ArrayList<String>());
				}
				treeMap.get(freq).add(word);
			}
		}
		
		int idx = 1;
		for(int freq: treeMap.descendingKeySet())
		{
			for(String word: treeMap.get(freq))
			{
				vocabMap.put(word, idx);
				idx++;
			}
		}
	}
	
	public static void dumpVocab(HashMap<String, Integer> hashMap, 
			String outputFile, 
			String encoding)
	{
		TreeMap<Integer, String> treeMap = new TreeMap<Integer, String>();
		for(String word: hashMap.keySet())
		{
			treeMap.put(hashMap.get(word), word);
		}
		
		try{
			PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(outputFile), encoding)));
			for(int idx: treeMap.keySet())
			{
				writer.write(treeMap.get(idx) + " " + idx + "\n");
			}
			writer.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static double cosineSim(String[] words1, String[] words2)
	{
		double sim = 0.0;
		HashSet<String> localWordVocab = new HashSet<String>();
		for(String word: words1)
		{
			localWordVocab.add(word);
		}
		for(String word: words2)
		{
			localWordVocab.add(word);
		}
		
		List<String> vocab = new ArrayList<String>();
		vocab.addAll(localWordVocab);
		
		Double[] idx1 = new Double[vocab.size()];
		Double[] idx2 = new Double[vocab.size()];
		
		for(int i = 0; i < vocab.size(); i++)
		{
			idx1[i] = 0.0;
			idx2[i] = 0.0;
		}
		
		for(String word1: words1)
		{
			idx1[vocab.indexOf(word1)] = 1.0;
		}
		
		for(String word2: words2)
		{
			idx2[vocab.indexOf(word2)] = 1.0;
		}
		
		return cosineSim(idx1, idx2);
	}
	
	public static Double cosineSim(Double[] value1, Double[] value2)
	{
		Double xx = 0.0;
		Double yy = 0.0;
		Double xy = 0.0;
		
		for(int i = 0; i < value1.length; i++)
		{
			Double x = value1[i];
			Double y = value2[i];
				
			xx = xx + x * x;
			yy = yy + y * y;
			xy = xy + x * y;
		}
		
		Double sim = xy / (Math.sqrt(xx) * Math.sqrt(yy));
		return sim;
	}
	
	public static int[] fillSentence(
			String[] words,
			HashMap<String, Integer> vocabMap)
	{
		List<Integer> idxList = new ArrayList<Integer>();
		
		for(int i = 0; i < words.length; i++)
		{
			String word = words[i];
			
			if(vocabMap.containsKey(word))
			{
				idxList.add(vocabMap.get(word));
			}
			else
			{
			}
		}
		int[] wordIns = new int[idxList.size()];
		for(int k = 0; k < idxList.size(); k++)
		{
			wordIns[k] = idxList.get(k);
		}
		
		return wordIns;
	}
	
}

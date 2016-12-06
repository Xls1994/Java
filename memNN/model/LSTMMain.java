package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import dataprepare.Data;
import dataprepare.Funcs;
import dataprepare.IO;
import duyuNN.*;
import duyuNN.combinedLayer.*;
import evaluationMetric.Metric;

public class LSTMMain {
	
	LookupLayer seedLookup;
	
	SimplifiedLSTMLayer seedLSTMCell;
	
	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;
	
	HashMap<String, Integer> wordVocab = null;
	
	public LSTMMain(
				String embeddingFile, 
				int embeddingLength,
				int classNum,
				String trainFile,
				String testFile,
				double randomizeBase,
				boolean isNormLookup) throws Exception
	{
		HashSet<String> wordSet = new HashSet<String>();
		loadData(trainFile, testFile, wordSet);		
		
		wordVocab = new HashMap<String, Integer>();
		double[][] table = Funcs.loadEmbeddingFile(embeddingFile, embeddingLength, "utf8", 
				isNormLookup, wordVocab, wordSet);
		
		seedLookup = new LookupLayer(embeddingLength, wordVocab.size(), 1);
		seedLookup.setEmbeddings(table);
		
		Random rnd = new Random(); 
		seedLSTMCell = new SimplifiedLSTMLayer(embeddingLength);
		seedLSTMCell.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		
		linearForSoftmax = new LinearLayer(embeddingLength, classNum);
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		
		softmax = new SoftmaxLayer(classNum);
		linearForSoftmax.link(softmax);
	}
	
	List<Data> trainDataList;
	List<Data> testDataList;  
	
	public void loadData(
			String trainFile,
			String testFile,
			HashSet<String> wordSet)
	{
		System.out.println("================ start loading corpus ==============");
		
		trainDataList =  Funcs.loadCorpus(trainFile, "utf8");
		testDataList = Funcs.loadCorpus(testFile, "utf8");

		List<Data> allList = new ArrayList<Data>();
		allList.addAll(trainDataList);
		allList.addAll(testDataList);
		
		for(Data data: allList)
		{
			String[] words = data.text.split(" ");
			for(String word: words)
			{
				if(word.equals("$t$"))
					continue;
				
				wordSet.add(word);
			}
			
			String[] targets = data.target.split(" ");
			for(String target: targets)
			{
				wordSet.add(target);
			}
		}
			
		System.out.println("training size: " + trainDataList.size());
		System.out.println("testDataList size: " + testDataList.size());
		System.out.println("wordSet.size: " + wordSet.size());
		
		System.out.println("================ finsh loading corpus ==============");
	}
	
	public void run(
			int roundNum,
			double clippingThreshold,
			double learningRate,
			int classNum,
			String dumpResultFile
			) throws Exception
	{
		double lossV = 0.0;
		int lossC = 0;
		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");
			Collections.shuffle(trainDataList, new Random());
			System.out.println("Finish shuffling training data.");
			
			for(int idxData = 0; idxData < trainDataList.size(); idxData++)
			{
				Data data = trainDataList.get(idxData);
				
				String text = data.text.replace("$t$", data.target);
				if(text.contains("$t$"))
				{
					System.err.println(data.text);
				}
				
				String[] words = text.split(" ");
				
				int[] wordIds = Funcs.fillSentence(words, wordVocab);
				
				SentenceLSTM sentLSTM = new SentenceLSTM(
						wordIds,
						seedLookup,
						seedLSTMCell);
				
				// link. important.
				sentLSTM.link(linearForSoftmax);
				
 				sentLSTM.forward();
 				linearForSoftmax.forward();
				softmax.forward();
				
				// set cross-entropy error 
				int goldPol = data.goldPol;
				lossV += -Math.log(softmax.output[goldPol]);
				lossC += 1;
				
				for(int k = 0; k < softmax.outputG.length; k++)
					softmax.outputG[k] = 0.0;
				softmax.outputG[goldPol] = 1.0 / softmax.output[goldPol];
				
				// if ||g|| >= threshold, then g <- g * threshold / ||g|| 
				if(Math.abs(softmax.outputG[goldPol]) > clippingThreshold)
				{
					if(softmax.outputG[goldPol] > 0)
						softmax.outputG[goldPol] =  clippingThreshold;
					else
						softmax.outputG[goldPol] =  -1.0 * clippingThreshold;
				}
				
				// backward
				softmax.backward();
				linearForSoftmax.backward();
				sentLSTM.backward();
				
				// update
				linearForSoftmax.update(learningRate);
				sentLSTM.update(learningRate);
				
				// clearGrad
				sentLSTM.clearGrad();
				linearForSoftmax.clearGrad();
				softmax.clearGrad();

				if(idxData % 500 == 0)
				{
					System.out.println("running idxData = " + idxData + "/" + trainDataList.size() + "\t "
							+ "lossV/lossC = " + String.format("%.4f", lossV) + "/" + lossC + "\t"
							+ " = " + String.format("%.4f", lossV/lossC)
							+ "\t" + new Date().toLocaleString());
				}
			}
			
			System.out.println("============= finish training round: " + round + " ==============");
			predict(round, dumpResultFile);
		}
	}
//	
	public void predict(int round,
			String dumpResultFile) throws Exception
	{
		System.out.println("=========== predicting round: " + round + " ===============");
		
		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();
		
		for(int idxData = 0; idxData < testDataList.size(); idxData++)
		{
			Data data = testDataList.get(idxData);
			
			String text = data.text.replace("$t$", data.target);
			if(text.contains("$t$"))
			{
				System.err.println(data.text);
			}
			
			String[] words = text.split(" ");
			
			int[] wordIds = Funcs.fillSentence(words, wordVocab);
			
			SentenceLSTM sentLSTM = new SentenceLSTM(
					wordIds,
					seedLookup,
					seedLSTMCell);
			
			// link. important.
			sentLSTM.link(linearForSoftmax);
			
			sentLSTM.forward();
			linearForSoftmax.forward();
			softmax.forward();
			
			int predClass = -1;
			double maxPredProb = -1.0;
			for(int ii = 0; ii < softmax.length; ii++)
			{
				if(softmax.output[ii] > maxPredProb)
				{
					maxPredProb = softmax.output[ii];
					predClass = ii;
				}
			}
			
			predList.add(predClass);
			goldList.add(data.goldPol);
			
			// clearGrad
			sentLSTM.clearGrad();
			linearForSoftmax.clearGrad();
			softmax.clearGrad();
		}
		
		Metric.calcMetric(goldList, predList);
		
		IO.writeIntFile(dumpResultFile + "-" + round + "-gold.txt", goldList, "utf8");
		IO.writeIntFile(dumpResultFile + "-" + round + "-pred.txt", predList, "utf8");
		
		System.out.println("============== finish predicting =================");
	}

	public static void main(String[] args) throws Exception
	{
		HashMap<String, String> argsMap = Funcs.parseArgs(args);
		
		System.out.println("==== begin configuration ====");
		for(String key: argsMap.keySet())
		{
			System.out.println(key + "\t\t" + argsMap.get(key));
		}
		System.out.println("==== end configuration ====");
		
		int embeddingLength = Integer.parseInt(argsMap.get("-embeddingLength"));
		String embeddingFile = argsMap.get("-embeddingFile");
		// windowsize = 1, 2 and 3 works well 
		int classNum = Integer.parseInt(argsMap.get("-classNum"));
		
		int roundNum = Integer.parseInt(argsMap.get("-roundNum"));
		double clippingThreshold = Double.parseDouble(argsMap.get("-clippingThreshold"));
		double learningRate = Double.parseDouble(argsMap.get("-learningRate"));
		double randomizeBase = Double.parseDouble(argsMap.get("-randomizeBase"));
		boolean isNormLookup = Boolean.parseBoolean(argsMap.get("-isNormLookup"));
		
		String trainFile = argsMap.get("-trainFile");
		String testFile  = argsMap.get("-testFile");
		
		String dumpResultFile = argsMap.get("-dumpResultFile");
		
		LSTMMain main = new LSTMMain(
				embeddingFile, 
				embeddingLength, 
				classNum, 
				trainFile, 
				testFile,
				randomizeBase,
				isNormLookup);
		
		main.run(roundNum, 
				clippingThreshold, 
				learningRate, 
				classNum,
				dumpResultFile);
	}
}

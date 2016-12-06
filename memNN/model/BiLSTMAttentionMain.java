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
import duyuNN.*;
import duyuNN.combinedLayer.*;
import evaluationMetric.Metric;

public class BiLSTMAttentionMain {
	
	LookupLayer seedLookup;
	
	SimplifiedLSTMLayer seedLSTMCellForw;
	SimplifiedLSTMLayer seedLSTMCellBack;
	
	ConnectLinearTanh attentionCellSeed;
	
	MultiConnectLayer connect;
	
	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;
	
	HashMap<String, Integer> wordVocab = null;
	
	public BiLSTMAttentionMain(
				String embeddingFile, 
				int embeddingLength,
				int classNum,
				String trainFile,
				String testFile,
				double randomizeBase,
				boolean isNormLookup,
				double attentionRandomBase) throws Exception
	{ 
		HashSet<String> wordSet = new HashSet<String>();
		loadData(trainFile, testFile, wordSet);		
		
		wordVocab = new HashMap<String, Integer>();
		double[][] table = Funcs.loadEmbeddingFile(embeddingFile, embeddingLength, "utf8", 
				isNormLookup, wordVocab, wordSet);
		
		seedLookup = new LookupLayer(embeddingLength, wordVocab.size(), 1);
		seedLookup.setEmbeddings(table);
		
		Random rnd = new Random(); 
		seedLSTMCellForw = new SimplifiedLSTMLayer(embeddingLength);
		seedLSTMCellForw.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		
		seedLSTMCellBack = new SimplifiedLSTMLayer(embeddingLength);
		seedLSTMCellBack.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		
		connect = new MultiConnectLayer(new int[]{embeddingLength, embeddingLength});
		
		linearForSoftmax = new LinearLayer(embeddingLength * 2, classNum);
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		
		softmax = new SoftmaxLayer(classNum);

		connect.link(linearForSoftmax);
		linearForSoftmax.link(softmax);
		
		attentionCellSeed = new ConnectLinearTanh(embeddingLength, embeddingLength, 1);
		
		attentionCellSeed.randomize(rnd, -1.0 * attentionRandomBase, attentionRandomBase);
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
			double attentionLearningRate
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
				
				String text = data.text;
				int targetIdx = text.indexOf("$t$");
				
				String forwText = text.substring(0, targetIdx + 3);
				String backText = text.substring(targetIdx);
				
				forwText = forwText.replace("$t$", data.target);
				backText = backText.replace("$t$", data.target);
				
				String[] forwWords = forwText.split(" ");
				String[] tmpBackWords = backText.split(" ");
				
				String[] backWords = new String[tmpBackWords.length];
				for(int i = 0; i < backWords.length; i++)
				{
					backWords[i] = tmpBackWords[tmpBackWords.length - 1 - i];
				}
				
				int[] forwWordIds = Funcs.fillSentence(forwWords, wordVocab);
				int[] backWordIds = Funcs.fillSentence(backWords, wordVocab);
				
				// target word vec
				double[] targetVec = new double[seedLookup.embeddingLength];
				String[] targetWords = data.target.split(" ");
				int[] targetIds = Funcs.fillSentence(targetWords, wordVocab);
				
				if(targetIds.length == 0)
				{
					System.err.println("targetIds.length == 0");
					continue;
				}
				
				for(int id: targetIds)
				{
					double[] xVec = seedLookup.table[id];
					for(int i = 0; i < xVec.length; i++)
					{
						targetVec[i] += xVec[i];
					}
				}
				for(int i = 0; i < targetVec.length; i++)
				{
					targetVec[i] = targetVec[i] / targetIds.length;
				}
				
				SentenceLSTMAttention sentLSTMForw = new SentenceLSTMAttention(
						forwWordIds,
						seedLookup,
						seedLSTMCellForw,
						attentionCellSeed,
						targetVec);
				
				SentenceLSTMAttention sentLSTMBack = new SentenceLSTMAttention(
						backWordIds,
						seedLookup,
						seedLSTMCellBack,
						attentionCellSeed,
						targetVec);
				
				if(sentLSTMForw.tanhList.size() == 0 || sentLSTMBack.tanhList.size() == 0)
				{
					System.err.println(data.text);
					continue;
				}
				
				// link. important.
				sentLSTMForw.link(connect, 0);
				sentLSTMBack.link(connect, 1);
				
				sentLSTMForw.forward();
				sentLSTMBack.forward();
				connect.forward();
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
				connect.backward();
				sentLSTMForw.backward();
				sentLSTMBack.backward();
				
				// update
				linearForSoftmax.update(learningRate);
				sentLSTMForw.update(learningRate, attentionLearningRate);
				sentLSTMBack.update(learningRate, attentionLearningRate);
				
				// clearGrad
				sentLSTMForw.clearGrad();
				sentLSTMBack.clearGrad();
				connect.clearGrad();
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
			predict(round);
		}
	}
//	
	public void predict(int round) throws Exception
	{
		System.out.println("=========== predicting round: " + round + " ===============");
		
		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();
		
		for(int idxData = 0; idxData < testDataList.size(); idxData++)
		{
			Data data = testDataList.get(idxData);
			
			String text = data.text;
			int targetIdx = text.indexOf("$t$");
			
			String forwText = text.substring(0, targetIdx + 3);
			String backText = text.substring(targetIdx);
			
			forwText = forwText.replace("$t$", data.target);
			backText = backText.replace("$t$", data.target);
			
			String[] forwWords = forwText.split(" ");
			String[] tmpBackWords = backText.split(" ");
			
			String[] backWords = new String[tmpBackWords.length];
			for(int i = 0; i < backWords.length; i++)
			{
				backWords[i] = tmpBackWords[tmpBackWords.length - 1 - i];
			}
			
			int[] forwWordIds = Funcs.fillSentence(forwWords, wordVocab);
			int[] backWordIds = Funcs.fillSentence(backWords, wordVocab);
			
			// target word vec
			double[] targetVec = new double[seedLookup.embeddingLength];
			String[] targetWords = data.target.split(" ");
			int[] targetIds = Funcs.fillSentence(targetWords, wordVocab);
			
			if(targetIds.length == 0)
			{
				System.err.println("targetIds.length == 0");
				continue;
			}
			
			for(int id: targetIds)
			{
				double[] xVec = seedLookup.table[id];
				for(int i = 0; i < xVec.length; i++)
				{
					targetVec[i] += xVec[i];
				}
			}
			for(int i = 0; i < targetVec.length; i++)
			{
				targetVec[i] = targetVec[i] / targetIds.length;
			}
			
			SentenceLSTMAttention sentLSTMForw = new SentenceLSTMAttention(
					forwWordIds,
					seedLookup,
					seedLSTMCellForw,
					attentionCellSeed,
					targetVec);
			
			SentenceLSTMAttention sentLSTMBack = new SentenceLSTMAttention(
					backWordIds,
					seedLookup,
					seedLSTMCellBack,
					attentionCellSeed,
					targetVec);
			
			if(sentLSTMForw.tanhList.size() == 0 || sentLSTMBack.tanhList.size() == 0)
			{
				System.err.println(data.text);
				continue;
			}
			
			// link. important.
			sentLSTMForw.link(connect, 0);
			sentLSTMBack.link(connect, 1);
			
			sentLSTMForw.forward();
			sentLSTMBack.forward();
			connect.forward();
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
			sentLSTMForw.clearGrad();
			sentLSTMBack.clearGrad();
			connect.clearGrad();
			linearForSoftmax.clearGrad();
			softmax.clearGrad();
		}
		
		Metric.calcMetric(goldList, predList);
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
		
		double attentionRandomBase = Double.parseDouble(argsMap.get("-attentionRandomBase"));
		double attentionLearningRate = Double.parseDouble(argsMap.get("-attentionLearningRate"));
		
		String trainFile = argsMap.get("-trainFile");
		String testFile  = argsMap.get("-testFile");
		
		BiLSTMAttentionMain main = new BiLSTMAttentionMain(
				embeddingFile, 
				embeddingLength, 
				classNum, 
				trainFile, 
				testFile,
				randomizeBase,
				isNormLookup,
				attentionRandomBase);
		
		main.run(roundNum, 
				clippingThreshold, 
				learningRate, 
				classNum,
				attentionLearningRate);
	}
}

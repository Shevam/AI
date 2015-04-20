import java.io.File;
import java.io.IOException;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.classification.Classifier;

import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.KDtreeKNN;
import net.sf.javaml.classification.AbstractClassifier;
import net.sf.javaml.classification.AbstractMeanClassifier;
import net.sf.javaml.classification.MeanFeatureVotingClassifier;
import net.sf.javaml.classification.NearestMeanClassifier;
import net.sf.javaml.classification.SOM;
import net.sf.javaml.classification.ZeroR;

import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.bayes.KDependentBayesClassifier;
import net.sf.javaml.classification.bayes.AbstractBayesianClassifier;

import net.sf.javaml.classification.tree.RandomTree;
import net.sf.javaml.classification.tree.RandomForest;

import net.sf.javaml.classification.evaluation.PerformanceMeasure;

//@SuppressWarnings("unused")
public class Main {
	public enum classifier { KNN, NB, RT };
	
	public static void main (String args[]) throws IOException
	{
		classifier method;
		String trainingset, testingset;
		
		trainingset = "data/census.train";
		testingset = "data/census3.test";
		method = classifier.NB;
		
		switch (method) {
			case KNN:
				KNNClassifier(trainingset, testingset);
				break;
			case NB:
				NaiveBayesClassifier(trainingset, testingset);
				break;
			case RT:
				RandomTree(trainingset, testingset);
				break;
			
			default:
				break;
		}
	}
	
	public static void KNNClassifier(String trainPath, String testPath) throws IOException 
	{
		Dataset trainingset, testingset;
		Classifier knn;
		int correct, wrong;
		Object predictedClassValue, realClassValue;
		
		System.out.println("kNN Classifier");
		trainingset = FileHandler.loadDataset(new File(trainPath), 13, " ");
		knn = new KNearestNeighbors(5);
		knn.buildClassifier(trainingset);
		
		testingset = FileHandler.loadDataset(new File(testPath), 13, " ");
		
		correct = 0;
		wrong = 0;
		for (Instance inst : testingset) {
			predictedClassValue = knn.classify(inst);
			realClassValue = inst.classValue();
			
			System.out.println(predictedClassValue + ":" +realClassValue);
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}
	
	public static void NaiveBayesClassifier(String trainPath, String testPath) throws IOException 
	{
		Dataset trainingset, testingset;
		Classifier nbc;
		int correct, wrong;
		Object predictedClassValue, realClassValue;
		
		System.out.println("Naive Bayes Classifier");
		trainingset = FileHandler.loadDataset(new File(trainPath), 13, " ");
		nbc = new NaiveBayesClassifier(true, true, false);
		nbc.buildClassifier(trainingset);
		
		testingset = FileHandler.loadDataset(new File(testPath), 13, " ");
		
		correct = 0;
		wrong = 0;
		for (Instance inst : testingset) {
			predictedClassValue = nbc.classify(inst);
			realClassValue = inst.classValue();
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}
	
	public static void RandomTree(String trainPath, String testPath) throws IOException 
	{
		Dataset trainingset, testingset;
		Classifier rt;
		int correct, wrong, i;
		Object predictedClassValue, realClassValue;
		Instance inst;
		
		System.out.println("Random Tree Classifier");
		trainingset = FileHandler.loadDataset(new File(trainPath), 13, " ");
		rt = new RandomTree(13, null);
		rt.buildClassifier(trainingset);
		
		testingset = FileHandler.loadDataset(new File(testPath), 13, " ");
		
		correct = 0;
		wrong = 0;
		i = 0;
		while (true) {
			inst = testingset.instance(i);
			if(inst==null||i>20)
				break;
			predictedClassValue = rt.classify(inst);
			realClassValue = inst.classValue();
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}
}

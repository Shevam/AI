import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.sampling.Sampling;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.KDtreeKNN;
import net.sf.javaml.classification.MeanFeatureVotingClassifier;
import net.sf.javaml.classification.NearestMeanClassifier;
import net.sf.javaml.classification.SOM;
import net.sf.javaml.classification.ZeroR;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.bayes.KDependentBayesClassifier; // Gives error
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.classification.meta.Bagging;
import net.sf.javaml.classification.meta.SimpleBagging;
import net.sf.javaml.classification.tree.RandomTree;
import net.sf.javaml.classification.tree.RandomForest;
import net.sf.javaml.clustering.SOM.GridType;
import net.sf.javaml.clustering.SOM.LearningType;
import net.sf.javaml.clustering.SOM.NeighbourhoodFunction;

public class Main {
	public enum classifiers {
		KNearestNeighbors, KDtreeKNN, MeanFeatureVotingClassifier, NearestMeanClassifier,
		SOM, ZeroR, NaiveBayesClassifier, KDependentBayesClassifier, Bagging, SimpleBagging,
		RandomTree, RandomForest
	};
	
	public static void main(String args[]) throws IOException {
		String trainingSetPath = "data/iris.train";
		String testingSetPath = "data/iris.test";
		int noOfAttributes = 4;
		int classValueIndex = 4;
		String fieldSeparator = ",";
		ArrayList<classifiers> listOfClassifierToRun;
		
		Dataset trainingSet = FileHandler.loadDataset(new File(trainingSetPath), classValueIndex, fieldSeparator);
		Dataset testingSet = FileHandler.loadDataset(new File(testingSetPath), classValueIndex, fieldSeparator);
		
		listOfClassifierToRun = new ArrayList<classifiers>();
		
		listOfClassifierToRun.add(classifiers.KNearestNeighbors);
		//listOfClassifierToRun.add(classifiers.KDtreeKNN);
		//listOfClassifierToRun.add(classifiers.MeanFeatureVotingClassifier);
		//listOfClassifierToRun.add(classifiers.NearestMeanClassifier);
		//listOfClassifierToRun.add(classifiers.SOM);
		//listOfClassifierToRun.add(classifiers.ZeroR);
		listOfClassifierToRun.add(classifiers.NaiveBayesClassifier);
		//listOfClassifierToRun.add(classifiers.KDependentBayesClassifier);
		//listOfClassifierToRun.add(classifiers.Bagging);
		//listOfClassifierToRun.add(classifiers.SimpleBagging);
		//listOfClassifierToRun.add(classifiers.RandomTree);
		listOfClassifierToRun.add(classifiers.RandomForest);
		
		for (classifiers classifierToUse : listOfClassifierToRun) {
			switch (classifierToUse) {
				case KNearestNeighbors:
					KNNClassifier(trainingSet, testingSet);
					break;
				case KDtreeKNN:
					KDtreeKNNClassifier(trainingSet, testingSet);
					break;
				case MeanFeatureVotingClassifier:
					MeanFeatureVotingClassifier(trainingSet, testingSet);
					break;
				case NearestMeanClassifier:
					NearestMeanClassifier(trainingSet, testingSet);
					break;
				case SOM:
					SOM(trainingSet, testingSet);
					break;
				case ZeroR:
					ZeroR(trainingSet, testingSet);
					break;
				case NaiveBayesClassifier:
					NaiveBayesClassifier(trainingSet, testingSet);
					break;
				case KDependentBayesClassifier:
					KDependentBayesClassifier(trainingSet, testingSet);
					break;
				case Bagging:
					Bagging(trainingSet, testingSet);
					break;
				case SimpleBagging:
					SimpleBagging(trainingSet, testingSet);
					break;
				case RandomTree:
					RandomTree(trainingSet, testingSet, noOfAttributes);
					break;
				case RandomForest:
					RandomForest(trainingSet, testingSet, noOfAttributes);
					break;
				default:
					System.err.println("Please add a classifier to listOfClassifierToRun");
			}
		}
	}

	public static void KNNClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier knn;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;
		
		System.out.println("KNN Classifier");
		
		knn = new KNearestNeighbors(5);
		knn.buildClassifier(trainingSet);
		
		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(knn, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = knn.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}
	
	public static void KDtreeKNNClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier kdtknn;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("KDtreeKNN Classifier");
		kdtknn = new KDtreeKNN(5);
		kdtknn.buildClassifier(trainingSet);
		
		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(kdtknn, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = kdtknn.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void MeanFeatureVotingClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier mfvc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Mean Feature Voting Classifier");
		mfvc = new MeanFeatureVotingClassifier();
		mfvc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(mfvc, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = mfvc.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void NearestMeanClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier nmc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Nearest Mean Classifier");
		nmc = new NearestMeanClassifier();
		nmc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(nmc, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = nmc.classify(inst);
			realClassValue = inst.classValue();

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void SOM(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier som;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("SOM Classifier");

		som = new SOM(2, 2, GridType.HEXAGONAL, 1000, 0.1, 8, LearningType.LINEAR, NeighbourhoodFunction.STEP);
		som.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(som, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = som.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void ZeroR(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier zr;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("ZeroR Classifier");

		zr = new ZeroR();
		zr.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(zr, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = zr.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void NaiveBayesClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier nbc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Naive Bayes Classifier");
		nbc = new NaiveBayesClassifier(true, true, false);
		nbc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(nbc, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
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
		System.out.println("------------------------------------------------------");
	}

	public static void KDependentBayesClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier kdbc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("KDependent Bayes Classifier");
		kdbc = new KDependentBayesClassifier(true, 5, new int[] { 2, 4, 6, 8 });
		kdbc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(kdbc, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = kdbc.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void RandomTree(Dataset trainingSet, Dataset testingSet, int noOfAttributes) throws IOException {
		Classifier rt;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Random Tree Classifier");
		rt = new RandomTree(noOfAttributes, null);
		rt.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(rt, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null) break;
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
		System.out.println("------------------------------------------------------");
	}
	
	public static void RandomForest(Dataset trainingSet, Dataset testingSet, int treeCount) throws IOException {
		Classifier rf;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Random Forest Classifier");
		rf = new RandomForest(treeCount);
		rf.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(rf, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null) break;
			predictedClassValue = rf.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}
	
	public static void Bagging(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier bg;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Bagging");
		bg = new Bagging(new Classifier[] { new KNearestNeighbors(5) }, Sampling.NormalBootstrapping, 0);
		bg.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(bg, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null) break;
			predictedClassValue = bg.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}
	
	public static void SimpleBagging(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier sb;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Simple Bagging");
		sb = new SimpleBagging(new Classifier[] { new KNearestNeighbors(5) });
		sb.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(sb, testingSet);
		for(Object classVariable:performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: "+classVariable);
			System.out.println("Number of true positive: "+pm.tp);
			System.out.println("Number of false positive: "+pm.fp);
			System.out.println("Number of true negative: "+pm.tn);
			System.out.println("Number of false negative: "+pm.fn);
			System.out.println("Accuracy: "+pm.getAccuracy());
			System.out.println("Correlation: "+pm.getCorrelation());
			System.out.println("Correlation Coefficient: "+pm.getCorrelationCoefficient());
			System.out.println("Error rate: "+pm.getErrorRate());
			System.out.println("F-Measure: "+pm.getFMeasure());
			System.out.println("Precision: "+pm.getPrecision());
			System.out.println("Recall: "+pm.getRecall());
			System.out.println("Cost: "+pm.getCost());
		}
		
		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null) break;
			predictedClassValue = sb.classify(inst);
			realClassValue = inst.classValue();
			
			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}
}

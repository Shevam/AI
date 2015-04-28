import java.io.File;
import java.io.IOException;
import java.util.Map;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KDtreeKNN;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.MeanFeatureVotingClassifier;
import net.sf.javaml.classification.NearestMeanClassifier;
import net.sf.javaml.classification.SOM;
import net.sf.javaml.classification.ZeroR;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.classification.meta.Bagging;
import net.sf.javaml.classification.meta.SimpleBagging;
import net.sf.javaml.classification.tree.RandomForest;
import net.sf.javaml.classification.tree.RandomTree;
import net.sf.javaml.clustering.SOM.GridType;
import net.sf.javaml.clustering.SOM.LearningType;
import net.sf.javaml.clustering.SOM.NeighbourhoodFunction;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.sampling.Sampling;
import net.sf.javaml.tools.data.FileHandler;

public class Main {
	public enum Classifiers {
		KNearestNeighbors, KDtreeKNN, MeanFeatureVotingClassifier, NearestMeanClassifier,
		SOM, ZeroR, NaiveBayesClassifier, KDependentBayesClassifier,
		Bagging, SimpleBagging, RandomTree, RandomForest
	};

	public static void main(String args[]) throws IOException {
		/* INPUT PARAMETERS */
		String trainingSetPath = "data/iris.train";
		String testingSetPath = "data/iris.test";
		int noOfAttributes = 4; // No of attributes per instance - Zero-based
		int classValueIndex = 4; // Zero-based index of class variable
		String fieldSeparator = ",";
		int k = 5; // For kNN and KDtreeKNN
		Classifiers[] classifiersToRun = new Classifiers[] {
		 Classifiers.KNearestNeighbors,
//		 Classifiers.KDtreeKNN,
//		 Classifiers.MeanFeatureVotingClassifier,
//		 Classifiers.NearestMeanClassifier,
//		 Classifiers.SOM,
//		 Classifiers.ZeroR,
//		 Classifiers.NaiveBayesClassifier,
//		 Classifiers.RandomTree,
//		 Classifiers.RandomForest,
//		 Classifiers.Bagging,
//		 Classifiers.SimpleBagging,
		};
		/* END - INPUT PARAMETERS */

		Dataset trainingSet = FileHandler.loadDataset(new File(trainingSetPath), classValueIndex, fieldSeparator);
		Dataset testingSet = FileHandler.loadDataset(new File(testingSetPath), classValueIndex, fieldSeparator);

		for (Classifiers classifierToUse : classifiersToRun) {
			switch (classifierToUse) {
			case KNearestNeighbors:
				KNNClassifier(trainingSet, testingSet, k);
				break;
			case KDtreeKNN:
				KDtreeKNNClassifier(trainingSet, testingSet, k);
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
			case RandomTree:
				RandomTree(trainingSet, testingSet, noOfAttributes);
				break;
			case RandomForest:
				RandomForest(trainingSet, testingSet, noOfAttributes);
				break;
			case Bagging:
				Bagging(trainingSet, testingSet);
				break;
			case SimpleBagging:
				SimpleBagging(trainingSet, testingSet);
				break;
			default:
				System.err.println("Please select classifiers for the array classifiersToRun");
			}
		}
	}

	public static void KNNClassifier(Dataset trainingSet, Dataset testingSet, int k) throws IOException {
		System.out.println("KNN Classifier");
		Classifier knn;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		knn = new KNearestNeighbors(k);
		knn.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(knn, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = knn.classify(inst);
			actualClassValue = inst.classValue();

			if (predictedClassValue.equals(actualClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
		System.out.println("------------------------------------------------------");
	}

	public static void KDtreeKNNClassifier(Dataset trainingSet, Dataset testingSet, int k) throws IOException {
		System.out.println("KDtreeKNN Classifier");
		Classifier kdtknn;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		kdtknn = new KDtreeKNN(k);
		kdtknn.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(kdtknn, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = kdtknn.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("Mean Feature Voting Classifier");
		Classifier mfvc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		mfvc = new MeanFeatureVotingClassifier();
		mfvc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(mfvc, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = mfvc.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("Nearest Mean Classifier");
		Classifier nmc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		nmc = new NearestMeanClassifier();
		nmc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(nmc, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = nmc.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("SOM Classifier");
		Classifier som;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		som = new SOM(2, 2, GridType.HEXAGONAL, 1000, 0.1, 8, LearningType.LINEAR, NeighbourhoodFunction.STEP);
		som.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(som, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = som.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("ZeroR Classifier");
		Classifier zr;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		zr = new ZeroR();
		zr.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(zr, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = zr.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("Naive Bayes Classifier");
		Classifier nbc;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		nbc = new NaiveBayesClassifier(true, true, false);
		nbc.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(nbc, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = nbc.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("Random Tree Classifier");
		Classifier rt;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		rt = new RandomTree(noOfAttributes, null);
		rt.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(rt, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null)
				break;
			predictedClassValue = rt.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("Random Forest Classifier");
		Classifier rf;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		rf = new RandomForest(treeCount);
		rf.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(rf, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null)
				break;
			predictedClassValue = rf.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		Object predictedClassValue, actualClassValue;

		System.out.println("Bagging");
		bg = new Bagging(new Classifier[] { new KNearestNeighbors(5) }, Sampling.NormalBootstrapping, 0);
		bg.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(bg, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null)
				break;
			predictedClassValue = bg.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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
		System.out.println("Simple Bagging");
		Classifier sb;
		PerformanceMeasure pm;
		int correct, wrong;
		Object predictedClassValue, actualClassValue;

		sb = new SimpleBagging(new Classifier[] { new KNearestNeighbors(5) });
		sb.buildClassifier(trainingSet);

		Map<Object, PerformanceMeasure> performanceMeasureMap = EvaluateDataset.testDataset(sb, testingSet);
		for (Object classVariable : performanceMeasureMap.keySet()) {
			pm = performanceMeasureMap.get(classVariable);
			System.out.println();
			System.out.println("Class variable: " + classVariable);
			System.out.println("Number of true positive: " + pm.tp);
			System.out.println("Number of false positive: " + pm.fp);
			System.out.println("Number of true negative: " + pm.tn);
			System.out.println("Number of false negative: " + pm.fn);
			System.out.println("Accuracy: " + pm.getAccuracy());
			System.out.println("Correlation: " + pm.getCorrelation());
			System.out.println("Correlation Coefficient: " + pm.getCorrelationCoefficient());
			System.out.println("Error rate: " + pm.getErrorRate());
			System.out.println("F-Measure: " + pm.getFMeasure());
			System.out.println("Precision: " + pm.getPrecision());
			System.out.println("Recall: " + pm.getRecall());
			System.out.println("Cost: " + pm.getCost());
		}

		System.out.println("\nOverall Performance");
		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			if (inst == null)
				break;
			predictedClassValue = sb.classify(inst);
			actualClassValue = inst.classValue();
			if (predictedClassValue.equals(actualClassValue))
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

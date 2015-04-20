import java.io.File;
import java.io.IOException;

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
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.classification.meta.Bagging;
import net.sf.javaml.classification.meta.SimpleBagging;
import net.sf.javaml.classification.tree.RandomTree;
import net.sf.javaml.classification.tree.RandomForest;
import net.sf.javaml.clustering.SOM.GridType;
import net.sf.javaml.clustering.SOM.LearningType;
import net.sf.javaml.clustering.SOM.NeighbourhoodFunction;

public class Main {
	public enum Classifiers {
		KNearestNeighbors, KDtreeKNN, MeanFeatureVotingClassifier, NearestMeanClassifier,
		SOM, ZeroR, NaiveBayesClassifier, KDependentBayesClassifier, Bagging, SimpleBagging,
		PerformanceMeasure, RandomTree, RandomForest
		
	};

	public static void main(String args[]) throws IOException {
		Classifiers methodToUse;
		String trainingSetPath = "data/iris.data";
		String testingSetPath = "data/iris.data";
		int noOfAttributes = 4;
		int classValueIndex = 4;
		String fieldSeparator = ",";

		Dataset trainingSet = FileHandler.loadDataset(new File(trainingSetPath), classValueIndex, fieldSeparator);
		Dataset testingSet = FileHandler.loadDataset(new File(testingSetPath), classValueIndex, fieldSeparator);

		methodToUse = Classifiers.SimpleBagging;

		switch (methodToUse) {
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
		case PerformanceMeasure:
			//PerformanceMeasure(trainingSet, testingSet);
			break;
		case RandomTree:
			RandomTree(trainingSet, testingSet, noOfAttributes);
			break;
		case RandomForest:
			RandomForest(trainingSet, testingSet, noOfAttributes);
			break;
		default:
			break;
		}
	}

	public static void KNNClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier knn;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("KNN Classifier");

		knn = new KNearestNeighbors(5);
		knn.buildClassifier(trainingSet);

		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = knn.classify(inst);
			realClassValue = inst.classValue();

			System.out.println(predictedClassValue + ":" + realClassValue);

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}

	public static void KDtreeKNNClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier kdtknn;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("KDtreeKNN Classifier");
		kdtknn = new KDtreeKNN(5);
		kdtknn.buildClassifier(trainingSet);

		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = kdtknn.classify(inst);
			realClassValue = inst.classValue();

			System.out.println(predictedClassValue + ":" + realClassValue);

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}

	public static void MeanFeatureVotingClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier mfvc;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Mean Feature Voting Classifier");
		mfvc = new MeanFeatureVotingClassifier();
		mfvc.buildClassifier(trainingSet);

		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = mfvc.classify(inst);
			realClassValue = inst.classValue();

			System.out.println(predictedClassValue + ":" + realClassValue);

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}

	public static void NearestMeanClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier nmc;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Nearest Mean Classifier");
		nmc = new NearestMeanClassifier();
		nmc.buildClassifier(trainingSet);

		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = nmc.classify(inst);
			realClassValue = inst.classValue();

			System.out.println(predictedClassValue + ":" + realClassValue);

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}

	public static void SOM(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier som;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("SOM Classifier");

		som = new SOM(2, 2, GridType.HEXAGONAL, 1000, 0.1, 8, LearningType.LINEAR, NeighbourhoodFunction.STEP);
		som.buildClassifier(trainingSet);

		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = som.classify(inst);
			realClassValue = inst.classValue();

			System.out.println(predictedClassValue + ":" + realClassValue);

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}

	public static void ZeroR(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier zr;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("ZeroR Classifier");

		zr = new ZeroR();
		zr.buildClassifier(trainingSet);

		correct = 0;
		wrong = 0;
		for (Instance inst : testingSet) {
			predictedClassValue = zr.classify(inst);
			realClassValue = inst.classValue();

			System.out.println(predictedClassValue + ":" + realClassValue);

			if (predictedClassValue.equals(realClassValue))
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct predictions: " + correct);
		System.out.println("Wrong predictions: " + wrong);
		System.out.println("Accuracy: " + (float) correct / (correct + wrong));
	}

	public static void NaiveBayesClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier nbc;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Naive Bayes Classifier");
		nbc = new NaiveBayesClassifier(true, true, false);
		nbc.buildClassifier(trainingSet);

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
	}

	public static void KDependentBayesClassifier(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier kdbc;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("KDependent Bayes Classifier");
		kdbc = new KDependentBayesClassifier(true, 5, new int[] { 2, 4, 6, 8 });
		kdbc.buildClassifier(trainingSet);

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
	}

	public static void RandomTree(Dataset trainingSet, Dataset testingSet, int noOfAttributes) throws IOException {
		Classifier rt;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Random Tree Classifier");
		rt = new RandomTree(noOfAttributes, null);
		rt.buildClassifier(trainingSet);

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
	}
	
	public static void RandomForest(Dataset trainingSet, Dataset testingSet, int treeCount) throws IOException {
		Classifier rf;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Random Forest Classifier");
		rf = new RandomForest(treeCount);
		rf.buildClassifier(trainingSet);

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
	}
	
	public static void Bagging(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier bg;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Bagging");
		bg = new Bagging(new Classifier[] { new KNearestNeighbors(5) }, Sampling.NormalBootstrapping, 0);
		bg.buildClassifier(trainingSet);

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
	}
	
	public static void SimpleBagging(Dataset trainingSet, Dataset testingSet) throws IOException {
		Classifier sb;
		int correct, wrong;
		Object predictedClassValue, realClassValue;

		System.out.println("Simple Bagging");
		sb = new SimpleBagging(new Classifier[] { new KNearestNeighbors(5) });
		sb.buildClassifier(trainingSet);

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
	}
}

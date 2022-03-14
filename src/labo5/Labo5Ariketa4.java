package labo5;

import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Labo5Ariketa4 {
	public static void main(String[] args) throws Exception {
		//train eta test-etik abiatuta
		
		//trainFSS --> Attribute selected classifier (FSS + NB) --> hace el "remove" inplícitamente [NO VIENE EN TODAS LAS LIBERÍAS]
		//evaluator + search --> FSS zehazteko
		//EN WEKA PONER "More options... --> Output Predictions
		
		
		////ARGUMENTUAK:
		
		//args[0] = train.arff: gainbegiratutako instantzien path (input) 					--> /home/jfu/Escritorio/EHES/Labo5/ariketa1/train.arff
		//args[1] = test_blind.arff: iragarpenan egiteko instantzien path (input) 			--> /home/jfu/Escritorio/EHES/Labo5/ariketa1/test_blind.arff
		//args[2] = test_predictions.txt: iragarpena gordetzeko fitxategiko path (output) 	--> /home/jfu/Escritorio/EHES/Labo5/ariketa3/test_predictions.arff
		
		////TRAIN ETA TEST MULTZOAK KARGATU
		DataSource sourceTrain = new DataSource(args[0]);
		Instances train = sourceTrain.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		DataSource sourceTest = new DataSource(args[1]);
		Instances test = sourceTest.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		////"ATTRIBUTE SELECTED CLASSIFIER" META-SAILKATZAILEA ERABILI FSS APLIKATZEKO ("train" multzoari eragin)
		//// ETA MULTZO BATERAGARRIAK LORTZEKO ("test" multzoari eragin)
		AttributeSelectedClassifier attSelCls = new AttributeSelectedClassifier();
		NaiveBayes nb = new NaiveBayes();
		attSelCls.setClassifier(nb);
		attSelCls.setEvaluator(new CfsSubsetEval());
		attSelCls.setSearch(new BestFirst());
		attSelCls.buildClassifier(train);

		////IRAGARPENA EGIN
		Evaluation ev = new Evaluation(train);
		FileWriter fw = new FileWriter(args[2]);
		String timeStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		fw.write(timeStamp + "\n\n");
		if(test.equalHeaders(train)) {
			System.out.println("TRAIN eta TEST multzoak bateragarriak dira");
			int i = 1;
			for(Instance instance : test) {
				int predictionInd = (int) ev.evaluateModelOnce(attSelCls, instance);
				String prediction = test.classAttribute().value(predictionInd);
				System.out.println(i + ": " + prediction);
				fw.write(i + ": " + prediction + "\n");
				i++;
			}
		}
		else {
			System.out.println("TRAIN eta TEST multzoak EZ dira bateragarriak");
		}
		fw.close();
		
		
	}

}

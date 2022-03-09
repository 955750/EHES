package labo5;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Labo5Ariketa1 {
	
	public static void main(String[] args) throws Exception {
		////ARGUMENTUAK: 

//	    data.arff: jatorrizko datuen path (input)
//	    train.arff datuak gordetzeko path (output)
//	    test_blind.arff datuak gordetzeko path (output)

		
		////DATUAK KARGATU
		DataSource source = new DataSource("/home/jfu/Descargas/data_supervised.arff");
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		
		////TRAIN eta TEST multzoak lortu
		
		//TRAIN multzoa lortu (%70) --> ERA GAINBEGIRATUAN
		Instances train = null;
		Instances unekoFold = null;
		StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
		srf.setNumFolds(10);
		for(int f = 1; f <= 7; f++) { //iterazio bakoitzean datuen %10 hartu (%10 * 7 = %70) 
			srf.setInputFormat(data);
			srf.setFold(f);
			srf.setInvertSelection(false);
			unekoFold = Filter.useFilter(data, srf);
			if(train == null) { //train daturik gabe dagoenean (1. iterazioan soilik)
				train = unekoFold;
			}
			else {
				for(int i = 0; i < unekoFold.numInstances(); i++) {
					train.add(unekoFold.get(i));
				}
			}
		}
		
		//TEST multzoa lortu (%30) --> ERA EZ-GAINBEGIRATUAN
		RemovePercentage rp = new RemovePercentage();
		rp.setInputFormat(data);
		rp.setInvertSelection(false);
		rp.setPercentage(70);
		Instances test = Filter.useFilter(data, rp);
		System.out.println(data.numInstances());
		System.out.println(train.numInstances());
		System.out.println(test.numInstances());
		
		//TEST multzoko instantzien klaseak '?' bihurtu
		for(int i = 0; i < test.numInstances(); i++) {
			test.instance(i).setClassMissing();
		}
		
//		
//		
//		
//		////SAILKATZAILEA --> NaiveBayes
//		NaiveBayes sailkatzailea = new NaiveBayes();
//		sailkatzailea.buildClassifier(train);
//		
//		
//		////EBALUATZAILEA
//		Evaluation ev = new Evaluation(train);
//		ev.evaluateModel(sailkatzailea, test);
//		
//		
//		////EMAITZAK ERAKUTSI
//		System.out.println(data.relationName());
//		Attribute a = data.attribute(0);
//		System.out.println(a.name());
//		System.out.println(a.type());
//		System.out.println(a.toString());
//		System.out.println(ev.toSummaryString());
//		System.out.println(ev.toClassDetailsString());
//		System.out.println(ev.toMatrixString("=== Nahasmen Matrizea ==="));
		
	}

}

package labo5;

import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Labo5Ariketa3 {
	
	public static void main(String[] args) throws Exception {
		
		////ARGUMENTUAK:
		 
		//args[0] = NB.model: ereduaren path (input) 										--> /home/jfu/Escritorio/EHES/Labo5/ariketa2/NB.model
		//args[1] = test_blind.arff: iragarpenan egiteko instantzien path (input) 			--> /home/jfu/Escritorio/EHES/Labo5/ariketa1/test_blind.arff
		//args[2] = trainFSS: atributuak ezabatuta dituen instantzien multzoa 				--> /home/jfu/Escritorio/EHES/Labo5/ariketa2/trainFSS.arff
		//args[3] = test_predictions.txt: iragarpena gordetzeko fitxategiko path (output) 	--> /home/jfu/Escritorio/EHES/Labo5/ariketa3/test_predictions.arff
		
		//TRAINFSS ETA TEST MULTZOAK KARGATU
		DataSource sourceTrain = new DataSource(args[2]);
		Instances trainFSS = sourceTrain.getDataSet();
		if (trainFSS.classIndex() == -1)
			trainFSS.setClassIndex(trainFSS.numAttributes() - 1);
		
		DataSource sourceTest = new DataSource(args[1]);
		Instances test = sourceTest.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		//EMANDAKO SAILKATZAILEA KARGATU
		Classifier cls = (Classifier) SerializationHelper.read(args[0]);
	
		//TEST MULTZOARI DAGOZKION ATRIBUTUAK EZABATU
		Remove rmFilter = new Remove();
		rmFilter.setInputFormat(trainFSS);
		rmFilter.setInvertSelection(false);
		test = Filter.useFilter(test, rmFilter);
		
		//IRAGARPENA EGIN
		Evaluation ev = new Evaluation(trainFSS);
		FileWriter fw = new FileWriter(args[3]);
		String timeStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		fw.write(timeStamp + "\n\n");
		if(test.equalHeaders(trainFSS)) {
			System.out.println("TRAIN eta TEST multzoak bateragarriak dira");
			System.out.println(trainFSS.numAttributes());
			System.out.println(test.numAttributes());
			int i = 1;
			System.out.println(ev.evaluateModelOnce(cls, test.firstInstance()));
			for(Instance instance : test) {
				int predictionInd = (int) ev.evaluateModelOnce(cls, instance);
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

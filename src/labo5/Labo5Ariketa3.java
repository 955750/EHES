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
		//args[2] = trainFSS: atributuak ezabatuta dituen instantzien multzoa (input)		--> /home/jfu/Escritorio/EHES/Labo5/ariketa2/trainFSS.arff
		//args[3] = test_predictions.txt: iragarpena gordetzeko fitxategiko path (output) 	--> /home/jfu/Escritorio/EHES/Labo5/ariketa3/test_predictions.arff
		//args[4] = testFSS: atributuak ezabatuta dituen instantzien multzoa (output)		--> /home/jfu/Escritorio/EHES/Labo5/ariketa3/testFSS.arff
		
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
		System.out.println("Atributu kopurua (Datuak aurreprozesatu baino lehen): " + test.numAttributes());
		Remove rmFilter = new Remove();
		rmFilter.setInvertSelection(false); //false = hautatutako zutabeak EZABATU; true = hautatutako zutabeak MANTENDU + gainontzekoak ezabatu
		rmFilter.setInputFormat(trainFSS);
		Instances testFSS = Filter.useFilter(test, rmFilter);
		System.out.println("Atributu kopurua (Datuak aurreprozesatu baino lehen): " + testFSS.numAttributes());
		System.out.println("Atributu kopurua (1. Instantzia): " + testFSS.firstInstance().numAttributes());
		
		//TESTFSS DATU SORTA GORDE EDUKIA IKUSTEKO
		ArffSaver saveTestFSS = new ArffSaver();
		saveTestFSS.setInstances(testFSS);
		saveTestFSS.setFile(new File(args[4]));
		saveTestFSS.writeBatch();
		
//		//IRAGARPENA EGIN
//		Evaluation ev = new Evaluation(trainFSS);
//		FileWriter fw = new FileWriter(args[3]);
//		String timeStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
//		fw.write(timeStamp + "\n\n");
//		if(testFSS.equalHeaders(trainFSS)) {
//			System.out.println("TRAIN eta TEST multzoak bateragarriak dira");
//			System.out.println(trainFSS.numAttributes());
//			System.out.println(testFSS.numAttributes());
//			int i = 1;
//			System.out.println(ev.evaluateModelOnce(cls, testFSS.firstInstance()));
//			for(Instance instance : testFSS) {
//				int predictionInd = (int) ev.evaluateModelOnce(cls, instance);
//				String prediction = testFSS.classAttribute().value(predictionInd);
//				System.out.println(i + ": " + prediction);
//				fw.write(i + ": " + prediction + "\n");
//				i++;
//			}
//		}
//		else {
//			System.out.println("TRAIN eta TEST multzoak EZ dira bateragarriak");
//		}
//		fw.close();
	}

}

package DataMiningExample;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;
import java.util.Scanner;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.AttributeStats;
//import weka.classifiers.lazy.IB1; Pakete hau ez da existitzen
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class TREC6Ariketa {
	
	public static void main(String[] args) throws Exception {
		//CSV datuak kargatu
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("/home/jfu/Descargas/Sample-Bulk-Recipient.csv"));
		Instances data = loader.getDataSet();
		
		//ARFF fitxategi batera pasa
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("/home/jfu/Descargas/trainCSV_ARFF.arff"));
		saver.writeBatch();
		
		//DATUAK KARGATU:
    	DataSource source = new DataSource("/home/jfu/Descargas/trainCSV_ARFF.arff");
    	Instances dataARFF = source.getDataSet();
		if (dataARFF.classIndex() == -1)
    		data.setClassIndex(data.numAttributes() - 1);
    	
    	//SAILKATZAILEA/ENTRENAMENDUA --> NaiveBayes
    	NaiveBayes nb = new NaiveBayes();
    	nb.buildClassifier(dataARFF);
    	
    	//EBALUATZAILEA --> 5-fCV
    	Evaluation ev = new Evaluation(dataARFF);
    	ev.crossValidateModel(nb, dataARFF, 5, new Random(1));
    	
    	//DATUAK ESKURATU
    	System.out.println(ev.toSummaryString());
    	System.out.println(ev.toClassDetailsString());
    	System.out.println(ev.toMatrixString());
	}
}

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
		loader.setSource(new File("/home/jfu/Descargas/train.csv"));
		Instances data = loader.getDataSet();
		
		//ARFF fitxategi batera pasa
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("/home/jfu/Descargas/trainCSV_ARFF.arff"));
		saver.writeBatch();
	}
}

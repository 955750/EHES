package DataMiningExample;

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
import weka.core.Attribute;
import weka.core.AttributeStats;
//import weka.classifiers.lazy.IB1; Pakete hau ez da existitzen
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.RemovePercentage;


public class holdOut {
	
	public static void main(String[] args) throws Exception {
		////DATUAK KARGATU
		DataSource source = new DataSource("/home/jfu/Descargas/adult.train.arff");
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		
		////DATUAK PRESTATU/FILTRATU (Hold-out aplikatu)
		
		// 1. RANDOMIZE
		data.randomize(new Random(1));
		
		// 2. SPLIT (false = 70 %; true = 30 %)
		RemovePercentage rp = new RemovePercentage();
		rp.setInputFormat(data);
		rp.setPercentage(70); //borras el 70% del principio --> te quedas con el 30 del final
		Instances test = Filter.useFilter(data, rp);
		rp.setInputFormat(data);
		rp.setInvertSelection(true);
		Instances train = Filter.useFilter(data, rp);
		
		
		////SAILKATZAILEA --> NaiveBayes
		NaiveBayes sailkatzailea = new NaiveBayes();
		sailkatzailea.buildClassifier(train);
		
		
		////EBALUATZAILEA
		Evaluation ev = new Evaluation(train);
		ev.evaluateModel(sailkatzailea, test);
		
		
		////EMAITZAK ERAKUTSI
		System.out.println(data.relationName());
		Attribute a = data.attribute(0);
		System.out.println(a.name());
		System.out.println(a.type());
		System.out.println(a.toString());
		System.out.println(ev.toSummaryString());
		System.out.println(ev.toClassDetailsString());
		System.out.println(ev.toMatrixString("=== Nahasmen Matrizea ==="));
	}
}

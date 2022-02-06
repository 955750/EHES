package DataMiningExample;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.lazy.IB1; Pakete hau ez da existitzen
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


public class labo2Xumea {

	public static void main(String[] args) throws Exception {
		//DATUAK KARGATU:
    	DataSource source = new DataSource("/home/jfu/Descargas/adult.train.arff");
    	Instances data = source.getDataSet();
    	if (data.classIndex() == -1)
    		data.setClassIndex(data.numAttributes() - 1);
    	
    	//SAILKATZAILEA/ENTRENAMENDUA --> NaiveBayes
    	NaiveBayes nb = new NaiveBayes();
    	nb.buildClassifier(data);
    	
    	//EBALUATZAILEA --> 5-fCV
    	Evaluation ev = new Evaluation(data);
    	ev.crossValidateModel(nb, data, 5, new Random(1));
    	
    	//DATUAK ESKURATU
    	System.out.println(ev.toSummaryString());
    	System.out.println(ev.toClassDetailsString());
    	System.out.println(ev.toMatrixString());
	}
}

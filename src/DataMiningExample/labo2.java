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
import weka.core.AttributeStats;
//import weka.classifiers.lazy.IB1; Pakete hau ez da existitzen
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


public class labo2 {

	public static void main(String[] args) throws Exception {
		System.out.println();
		System.out.println("-------------------------------------------");
		System.out.println();
		//DATUAK KARGATU:
		try {
			if(args.length != 2) {
				throw new ArrayIndexOutOfBoundsException();
			}
			
			DataSource source = null;
			try {
				source = new DataSource(args[0]);	
			} catch(IllegalArgumentException e) {
				System.out.println();
				System.out.println("ERROREA: Datu-fitxategiaren path-a berrikusi:" + args[0]);
				System.out.println();
			}
			
			Instances data = null;
			Evaluation ev = null;
			try {
				data = source.getDataSet();
				if (data.classIndex() == -1)
		    		data.setClassIndex(data.numAttributes() - 1);
    	
	    	
		    	//SAILKATZAILEA/ENTRENAMENDUA --> NaiveBayes
		    	NaiveBayes nb = new NaiveBayes();
		    	nb.buildClassifier(data);
		    	
		    	//EBALUATZAILEA --> 5-fCV
		    	ev = new Evaluation(data);
		    	ev.crossValidateModel(nb, data, 5, new Random(1));

	    	
		    	//DATUAK ESPORTATU
		    	try {
		            FileWriter fw = new FileWriter(args[1]);
		            //1. Exekuzio data
		            String timeStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		            fw.write(timeStamp + "\n\n");
		            //2. Fitxategiaren path-a
		            fw.write(args[1] + "\n\n");
		            //3. Nahasmen-matrizea
		            fw.write(ev.toMatrixString());
		            fw.close();
		        }
		        catch(IOException e) {
		        	System.out.println("ERROREA: Emaitza fitxategiaren path-a berrikusi: " + args[1]);
		        }
		    	
		    	//DATUAK INPRIMATU
		    	System.out.println("Instantzia kopurua: " + (int) ev.numInstances());
		    	System.out.println("Atributu kopurua: " + data.numAttributes());
		    	System.out.println("Lehenengo atributuak har ditzakeen balio ezberdinak: " + data.numDistinctValues(0));
		    	AttributeStats attStats = data.attributeStats(data.numAttributes() - 2);
		    	System.out.println("Azken aurreko atributuak dituen missing value kopurua: " + attStats.missingCount);
			} catch(NullPointerException e) {
				System.out.println("ERROREA: Datu-fitxategiaren edukia berrikusi: " + args[0]);			
			}
		} catch(ArrayIndexOutOfBoundsException e2) {
			System.out.println("ERROREA: Formatu desegokia.");
			System.out.println("Formatu egokia: java -jar labo2.jar /path/fitxIzena.arff /path/emaitzaFitxIzena.txt");
		}
		
	}
}

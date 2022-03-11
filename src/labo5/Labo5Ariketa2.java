package labo5;

import java.io.File;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Labo5Ariketa2 {
	
	public static void main(String[] args) throws Exception {
	
		////ARGUMENTUAK
		//args[0] = train.arff: gainbegiratutako instantzien path (input) --> /home/lsi/Descargas/data_supervised.arff
	    //args[1] = NB.model: eredua gordetzeko irteerako path (output) --> /home/lsi/Escritorio/NB.model
		//args[2] = trainFSS: atributuak ezabatuta dituen instantzien multzoa --> /home/lsi/Escritorio/trainFSS.arff

		////DATUAK KARGATU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
				
		
		////TRAINFSS multzoak lortu
		System.out.println("Atributu kopurua (Datuak aurreprozesatu baino lehen): " + data.numAttributes());
		AttributeSelection attSelFilter = new AttributeSelection();
		attSelFilter.setInputFormat(data);
		attSelFilter.setEvaluator(new CfsSubsetEval());
		attSelFilter.setSearch(new BestFirst());
		Instances trainFSS = Filter.useFilter(data, attSelFilter);
		System.out.println("Atributu kopurua (Datuak aurreprozesatu baino lehen): " + trainFSS.numAttributes());
		
		////EREDUA SORTU
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainFSS);
		
		////EREDUA ETA TRAINFSS MULTZOA ESPORTATU
		SerializationHelper.write(args[1], nb);		
		ArffSaver saveTrain = new ArffSaver();
		saveTrain.setInstances(trainFSS);
		saveTrain.setFile(new File(args[2]));
		saveTrain.writeBatch();
	}
}

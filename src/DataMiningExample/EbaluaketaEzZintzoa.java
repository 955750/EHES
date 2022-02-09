package DataMiningExample;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
public class EbaluaketaEzZintzoa {

	public static void main(String[] args) throws Exception{
		
		//DATUAK KARGATU (DATA = TRAIN = TEST)
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		//SAILKATZAILEA/ENTRENAMENDUA --> NaiveBayes
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(data);
		
		//EBALUATZAILEA
		Evaluation ev = new Evaluation(data);
		ev.evaluateModel(nb, data);
		
		//EMAITZAK ESKURATU
		System.out.println(ev.toSummaryString());
		System.out.println(ev.toClassDetailsString());
		System.out.println(ev.toMatrixString());
	}
}

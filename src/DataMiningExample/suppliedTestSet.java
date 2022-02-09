package DataMiningExample;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class suppliedTestSet {

	public static void main(String[] args) throws Exception{
		//DATUAK KARGATU
		DataSource trainS = new DataSource(args[0]);
		Instances train = trainS.getDataSet();
		if(train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		DataSource testS = new DataSource(args[1]);
		Instances test = testS.getDataSet();
		if(test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		//SAILKATZAILEA/ENTRENAMENDUA
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		
		//EBALUATZAILEA
		Evaluation ev = new Evaluation(train);
		ev.evaluateModel(nb, test);
		
		//EMAITZAK ESKURATU
		System.out.println(ev.toSummaryString());
		System.out.println(ev.toClassDetailsString());
		System.out.println(ev.toMatrixString());
		
	}
}

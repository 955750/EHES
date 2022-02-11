package DataMiningExample;

import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ZeroR_OneR_IBk {
	
	public static void main(String[] args) throws Exception{
		
		//DATUAK KARGATU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		//SAILKATZAILEA/ENTRENAMENDUA --> ZeroR, OneR, IBk
//		ZeroR zeroR = new ZeroR();
//		zeroR.buildClassifier(data);
//		OneR oneR = new OneR();
//		oneR.buildClassifier(data);
		IBk IBk = new IBk();
		IBk.buildClassifier(data);


		//EBALUATZAILEA
		Evaluation ev = new Evaluation(data);
		ev.evaluateModel(IBk, data);
		//ev.crossValidateModel(zeroR, data, 5, new Random(1));
		
		//DATUAK ESKURATU
		System.out.println(ev.toSummaryString());
		System.out.println(ev.toClassDetailsString());
		System.out.println(ev.toMatrixString());
	}
}

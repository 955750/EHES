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


///////////////////////////////////////////////////////
// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
///////////////////////////////////////////////////////
public class prueba2 {
	
    public static void main(String[] args) throws Exception {
		
    	//Datuak kargatu:
    	DataSource source = new DataSource("/home/jfu/Descargas/adult.train.arff");
    	Instances data = source.getDataSet();
    	if (data.classIndex() == -1)
    		data.setClassIndex(data.numAttributes() - 1);		
		

		//Sailkatzailea prestatu --> NaiveBayes:
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(data);
		
		
		//Ebaluatzailea prestatu --> 5-fCV
		Evaluation evaluator = new Evaluation(data);
		evaluator.crossValidateModel(nb, data, 5, new Random(1)); // Random(1): the seed=1 means "no shuffle" :-!
		
		
		//Datuak eskuratu
		double a = Utils.roundDouble(2.4654653, 3);
		int correct = (int) evaluator.correct();
		double pctCorrect = Utils.roundDouble(evaluator.pctCorrect(), 4);
		int incorrect = (int) evaluator.incorrect();
		double pctIncorrect = Utils.roundDouble(evaluator.pctIncorrect(), 4);
		double kappa = Utils.roundDouble(evaluator.kappa(), 4);
		double mae = Utils.roundDouble(evaluator.meanAbsoluteError(), 4);    
		double rmse = Utils.roundDouble(evaluator.rootMeanSquaredError(), 4);
		double rae = Utils.roundDouble(evaluator.relativeAbsoluteError(), 4);
		double rrse=Utils.roundDouble(evaluator.rootRelativeSquaredError(), 4);
		//double confMatrix[][]= evaluator.confusionMatrix(); (EDO HAU BUKLE BATEKIN INPRIMATU)
		String cf2 = evaluator.toMatrixString();
		int totalInstances = (int) evaluator.numInstances();
		String detailedAccuracyByClass = evaluator.toClassDetailsString();
		
		System.out.println("Correctly Classified Instances       " + correct + 
				"               " + pctCorrect + " %");
		System.out.println("Incorrectly Classified Instances      " + incorrect + 
				"               " + pctIncorrect + " %");
		System.out.println("Kappa statistic                          " + kappa);
		System.out.println("Mean absolute error                      " + mae);
		System.out.println("Root mean squared error                  " + rmse);
		System.out.println("Relative absolute error                 " + rae + " %");
		System.out.println("Root relative squared error             " + rrse + " %");	
		System.out.println("Total Number of Instances            " + totalInstances);
		System.out.println(cf2);
		System.out.println(evaluator.toSummaryString());
		System.out.println(detailedAccuracyByClass);
    }
}

package Labo3_4;

import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class Labo3CVParameterSelection {

	public static void main(String[] args) throws Exception {
		////DATUAK KARGATU + AURREPROZESAMENDUA
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
//		Randomize r = new Randomize();
//		r.setRandomSeed(3);
//		r.setInputFormat(data);
//		Instances dataR = Filter.useFilter(data, r);
		
			
		////SAILKATZAILEA
		CVParameterSelection paramSel = new CVParameterSelection();
		IBk IBk = new IBk();
		paramSel.setClassifier(IBk);
		

        ////BEHARREZKO HASIERAKETAK
        //k-ren mugak zehaztu (k = [1, numInstances / 4]; urratsa = [max / 10])
        int iterazioKop = 10;
        int max = data.numInstances() / 4; //max = 156
        int urratsa = max / iterazioKop;
        System.out.println(max);
        //1 16 31 46 61 76 91 106 121 136 151
        //paramSel.addCVParameter("K 1 151 11");
        int maxCVPar = 1 + urratsa * 10;
        paramSel.addCVParameter("K 1 " + maxCVPar + " 11");
        paramSel.setSeed(3);
        paramSel.setNumFolds(5);
        paramSel.buildClassifier(data);
        //System.out.println(paramSel.getCVParameters().length);;
        String[] a = paramSel.getBestClassifierOptions();
        for(int i = 0; i < a.length; i++) {
        	System.out.println(i + ": " + a[i]);
        }
       
        ////SAILKATZAILEARI PARAMETRO OPTIMOAK ESLEITU
        IBk iBkOPT = new IBk();
        int kOpt = Integer.valueOf(a[1]);
        iBkOPT.setKNN(kOpt);
        LinearNNSearch dOpt = new LinearNNSearch();
        dOpt.setDistanceFunction(new EuclideanDistance());
        iBkOPT.setNearestNeighbourSearchAlgorithm(dOpt);
        SelectedTag wOpt = new SelectedTag(iBkOPT.WEIGHT_NONE, iBkOPT.TAGS_WEIGHTING);
        iBkOPT.setDistanceWeighting(wOpt);
        ////EBALUAZIOA
        Evaluation evOPT = new Evaluation(data);
        evOPT.crossValidateModel(iBkOPT, data, 5, new Random(3));
        ////DATUAK ESKURATU
    	System.out.println(evOPT.toSummaryString());
    	System.out.println(evOPT.toClassDetailsString());
    	System.out.println(evOPT.toMatrixString());
        
//        //EBALUAZIOA (5-FCV; randomSeed = 3)
//        Evaluation ev = new Evaluation(data);
//        ev.crossValidateModel(paramSel, data, 10, new Random(1));
//        
//        //DATUAK ESKURATU
//    	System.out.println(ev.toSummaryString());
//    	System.out.println(ev.toClassDetailsString());
//    	System.out.println(ev.toMatrixString());
//    	System.out.println("\n--------------------------------------------------------------\n");
        
        
		
	}
}

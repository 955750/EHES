package DataMiningExample;

import java.util.ArrayList;
import java.util.Iterator;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class nIteraziokoHoldOut {
	
	//Metodo laguntzaileak
	private static Double batazbestekoa(ArrayList<Double> zerrenda) {
		Iterator<Double> itr = zerrenda.iterator();
		Double emaitza = 0.0;
		while(itr.hasNext()) {
			emaitza = emaitza + itr.next();
		}
		emaitza = emaitza / zerrenda.size();
		return emaitza;
	}
	
	private static Double desbiderazioTipikoa(ArrayList<Double> zerrenda, Double batazbestekoa) {
		//bariantza = SUM(xi²)/N -batbes²
		Iterator<Double> itr = zerrenda.iterator();
		Double bariantza = 0.0;
		while(itr.hasNext()) {
			bariantza = bariantza + Math.pow(itr.next(), 2);
		}
		bariantza = bariantza / zerrenda.size() - Math.pow(batazbestekoa, 2);
		Double stdev = Math.sqrt(bariantza);
		return stdev;
	}
	
	public static void main(String[] args) throws Exception{
		//args[0] = datu-fitxategiaren izena
		//args[1] = iterazio kopurua = N
		ArrayList<Double> correctInstances = new ArrayList<Double>();
		
		
		//DATUAK KARGATU
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		////DATUAK PRESTATU/FILTRATU --> N Iterazioko Hold-Out
		for(int i = 1; i <= Integer.valueOf(args[1]); i++) {
			//1. Randomize
			Randomize r = new Randomize();
			r.setRandomSeed(i);
			r.setInputFormat(data);
			Instances dataR = Filter.useFilter(data, r);
			
			//2. Split 
			RemovePercentage rp = new RemovePercentage();
			rp.setInputFormat(dataR);
			rp.setPercentage(70);
			Instances test = Filter.useFilter(dataR, rp);
			rp.setInputFormat(dataR);
			rp.setInvertSelection(true);
			Instances train = Filter.useFilter(dataR, rp);
			
			////SAILKATZAILEA/ENTRENAMENDUA --> NaiveBayes
			NaiveBayes sailkatzailea = new NaiveBayes();
			sailkatzailea.buildClassifier(train);
			
			////EBALUATZAILEA
			Evaluation ev = new Evaluation(train);
			ev.evaluateModel(sailkatzailea, test);
			
			////EMAITZA GORDE
			correctInstances.add(ev.pctCorrect());
			System.out.println(i + " " + ev.pctCorrect());
		}
		
		//ONDO IRAGARRITAKO INSTANTZIEN BATAZBESTEKOA ETA DES.TIP. KALKULATU
		System.out.println("\nBatazbestekoa: " + batazbestekoa(correctInstances));
		System.out.println("Desbiderazio tipikoa: " + desbiderazioTipikoa(correctInstances, batazbestekoa(correctInstances)));
		
	}
}

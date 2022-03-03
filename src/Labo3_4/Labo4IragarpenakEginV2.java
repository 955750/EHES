package Labo3_4;

import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class Labo4IragarpenakEginV2 {
	
	public static void main(String[] args) throws Exception {
		////2. PROGRAMA
		//args[0] = NB.model: eredua non dagoen esaten digun path (input)
		//args[1] = test_predictions.txt: iragarpena gordetzeko fitxategiko path (output)
		
		//DATUAK KARGATU
		DataSource source = new DataSource("/home/jfu/Descargas/data_supervised.arff");
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		//EMANDAKO SAILKATZAILEA KARGATU
		Classifier cls = (Classifier) SerializationHelper.read(args[0]);
		
		//IRAGARPENA EGIN ETA EMAITZA GORDE
		Evaluation ev = new Evaluation(data);
		FileWriter fw = new FileWriter(args[1]);
		String timeStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		fw.write(timeStamp + "\n\n");
		int i = 1;
		String zuzenaOkerra = null;
		int zuzenKopurua = 0;
		for(Instance instance : data) {
			int predictionInd = (int) ev.evaluateModelOnce(cls, instance);
			String actual = data.classAttribute().value((int) instance.classValue());
			String prediction = data.classAttribute().value(predictionInd);
			zuzenaOkerra = "OKERRA";
			if(actual.equals(prediction)) {
				zuzenaOkerra = "ZUZENA";
				zuzenKopurua++;
			}
			
			System.out.println(i + ".iterazioa: " + zuzenaOkerra + "\n" + "Benetako balioa:	" + actual + "		Iragarritako balioa: " + prediction);
			fw.write(i + ": " + prediction + "\n");
			i++;
		}
		//ONDO DOAN EGIAZTATZEKO
		double zuzenEhunekoa = zuzenKopurua * 100.0 / data.numInstances();
		System.out.println("\n\nBenetako zuzenen ehunekoa: " + ev.pctCorrect());
		System.out.println(zuzenKopurua);
		System.out.println("Iragarritako zuzenen ehunekoa: " + zuzenEhunekoa);
		fw.close();
//		system.out.println(ev.tosummarystring());
//		system.out.println(ev.toclassdetailsstring());
//		system.out.println(ev.tomatrixstring());
		
	}

}

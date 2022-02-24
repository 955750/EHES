package Labo3_4;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Labo3ParamEkorketa {

	public static void main(String[] args) throws Exception {
        
        //DATUAK KARGATU
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        if(data.classIndex() == - 1)
            data.setClassIndex(data.numAttributes() - 1);
        
        //Parametro optimoak gordetzeko zerrenda eta beharrezko hasieraketak
        ArrayList<String> paramOptimoak = new ArrayList<String>();
        int iterazioKop = 10;
        int urratsa = 1;
        int max = data.numInstances() / 4;
        double fMeasureMax = 0.0;
        
        //Klase minoritarioaren id-a lortu (KASU HONETAN --> B (ID = 1))
        int[] klaseMaiztasunak = data.attributeStats(data.classIndex()).nominalCounts;
        int klaseMinId = 0;
        int unekoMaiztasunMin = klaseMaiztasunak[0];
        for(int i = 1; i < klaseMaiztasunak.length; i++) {
        	if(klaseMaiztasunak[i] < unekoMaiztasunMin) {
        		unekoMaiztasunMin = klaseMaiztasunak[i];
        		klaseMinId = i;
        	}        		
        }
        
        //Auzokide kopurua = k (KNN)
        int kOpt = 0;
        //Metrika = d (nearestNeighbourSearchAlgorithm --> distanceFunction)
        String dOpt;
        //Distantziaren ponderazio faktorea = w (distanceWeighting)
        String wOpt;
        

        //10-FCV KNN-ren PARAM. EKORKETA EGITEKO
        //**IR HASTA NUMINSTANCES = ZEROR [NO ES ÓPTIMO // KONTZEPTUALKI EZ ZUZENA]
        for(int i = 1; i <= iterazioKop; i++) {
        	//bukle para métrica (BUSCAR VALORES POSIBLES)
        		//bucle para distanceWeighting (BUSCAR VALORES POSIBLES)
        			//klase minoritarioa lortu eta f-measure aztertu
        			//if x > currentOPTFMeasure --> balioOptimoak eguneratu
            //SAILKATZAILEA/ENTRENAMENDUA
            IBk iBk = new IBk();
            iBk.setKNN(urratsa);
            
            iBk.buildClassifier(data);
            
            //EBALUAZIOA
            Evaluation ev = new Evaluation(data);
            ev.crossValidateModel(iBk, data, 5, new Random(3));
            //DATUAK ESKURATU
        	System.out.println(ev.toSummaryString());
        	System.out.println(ev.toClassDetailsString());
        	System.out.println(ev.toMatrixString());
        	//System.out.println("KLASE MINORITARIOAREN ('B') F-MEASURE: " + ev.fMeasure(2));
        	//if x > currentOPTFMeasure --> balioOptimoak eguneratu
            
            //if(ev.fMeasure(classIndex))
            
            //PARAMETRO BERRIAK AUKERATU
            urratsa += max / iterazioKop; 
            
        }
    }

}

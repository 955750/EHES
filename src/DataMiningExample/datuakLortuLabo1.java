package DataMiningExample;

import java.util.Enumeration;
import java.util.Iterator;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class datuakLortuLabo1 {
	
	public static void main(String[] args) throws Exception{
		
		////DATUAK ESKURATU
		DataSource source = new DataSource("/home/jfu/Descargas/heart-c.arff");
		Instances data = source.getDataSet();
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		////ESKATUTAKO DATUAK ATERA
		
		//1. Klasearen balio posibleak
		Enumeration<Object> balioPos = data.attribute(data.classIndex()).enumerateValues();
		String balioPosibleak = "";
		while(balioPos.hasMoreElements()) {
			balioPosibleak += " " + balioPos.nextElement();
		}
		System.out.println("1. Klasearen balio posibleak: " + balioPosibleak + "\n");
		
		//2. Instantzia kopurua
		int instantziaKop = data.numInstances();
		System.out.println("2. Instantzia kopurua: " + instantziaKop + "\n");
		
		//3. Instantziak karakterizatzeko atributu kopurua
		int atributuKop = data.numAttributes();
		System.out.println("3. Atributu kopurua: " + atributuKop + "\n");
		
		//4. Lehenengo 5 atributuetarako
		
			//4.1 Atributu mota
		System.out.println("4.1 Atributu motak:");
		for(int i = 0; i <= 4; i++) {
			String izena = data.attribute(i).name();
			String mota = Attribute.typeToString(data.attribute(i));
			System.out.println("	" + izena + ": " + mota);
		}
		
			//4.2 Missing value kopurua (%)
		System.out.println("\n 4.2 Missing value kopurua (%): ");
		for(int i = 0; i <= 4; i++) {
			String izena = data.attribute(i).name();
			AttributeStats attStats = data.attributeStats(i);
			int misValKop = attStats.missingCount;
			int ehunekoa = misValKop / attStats.totalCount;
			System.out.println("	" + izena + ": " + misValKop + " (%" + ehunekoa + ")");
		}
		
			//4.3 Balio desberdin kopurua [DISTINCT VALUES]
		System.out.println("\n 4.3 Distinct value kopurua: ");
		for(int i = 0; i <= 4; i++) {
			String izena = data.attribute(i).name();
			AttributeStats attStats = data.attributeStats(i);
			int disValKop = attStats.distinctCount;
			System.out.println("	" + izena + ": " + disValKop);
		}
		
			//4.4 Unique value kopurua (%) [Behin bakarrik daudenak]
		System.out.println("\n 4.4 Unique value kopurua: ");
		for(int i = 0; i <= 4; i++) {
			String izena = data.attribute(i).name();
			AttributeStats attStats = data.attributeStats(i);
			int disValKop = attStats.uniqueCount;
			double ehunekoa = disValKop  * 100.0 / attStats.totalCount;
			System.out.println("	" + izena + ": " + disValKop + " (%" + ehunekoa + ")");
		}
		
			//4.5 Atributu numerikoen MIN, MAX, BATBES, DES.TIP
		System.out.println("\n 4.5 Atributu numerikoen MIN, MAX, BATBES, DES.TIP: ");
		for(int i = 0; i <= 4; i++) {
			String izena = data.attribute(i).name();
			AttributeStats attStats = data.attributeStats(i);
			Stats attNumStats = attStats.numericStats;
			if(attNumStats != null)
				System.out.println(izena + ": \n" + attNumStats);
		}
		
		
		
	}

}

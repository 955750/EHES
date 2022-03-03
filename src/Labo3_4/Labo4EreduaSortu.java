package Labo3_4;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Labo4EreduaSortu {
    
    public static void main(String[] args) throws Exception{        
        ////1. PROGRAMA
//        args[0] = data.arff: datuen path (input)
//        args[1] = NB.model: eredua gordetzeko path (output)
//        args[2] = KalitatearenEstimazioa.txt: kalitatearen estimazioa gordetzeko path (output)

        
        
        
        //DATUAK KARGATU:
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        
        //SAILKATZAILEA/ENTRENAMENDUA --> NaiveBayes
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);
        
        //1. Estimazioa --> (5-fCV)
        Evaluation ev1 = new Evaluation(data);
        ev1.crossValidateModel(nb, data, 5, new Random(1));
        System.out.println(ev1.toSummaryString());
        System.out.println(ev1.toClassDetailsString());
        System.out.println(ev1.toMatrixString());
        
        //2. Estimazioa --> Hold-out (%70)
        //DATUAK PRESTATU/FILTRATU (Hold-out aplikatu)
        
        // 1. RANDOMIZE
        Randomize r = new Randomize();
        r.setRandomSeed(3);
        r.setInputFormat(data);
        Instances dataR = Filter.useFilter(data, r);
        //data.randomize(new Random(3));
        
        // 2. SPLIT (false = 70 %; true = 30 %)
        RemovePercentage rp = new RemovePercentage();
        rp.setInputFormat(dataR);
        rp.setPercentage(70); //borras el 70% del principio --> te quedas con el 30 del final
        Instances test = Filter.useFilter(dataR, rp);
        rp.setInputFormat(dataR);
        rp.setInvertSelection(true);
        Instances train = Filter.useFilter(dataR, rp);
        
        //DATUAK ESKURATU (Hold-out %70)
        Evaluation ev2 = new Evaluation(data);
        ev2.evaluateModel(nb, test);
        System.out.println(ev2.toSummaryString());
        System.out.println(ev2.toClassDetailsString());
        System.out.println(ev2.toMatrixString());
        
        
        ///EREDUA GORDE
        SerializationHelper.write(args[1], nb);
        
        //DATUAK ESPORTATU
        try {
            FileWriter fw = new FileWriter(args[2]);
            //1. Exekuzio data
            String timeStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
            fw.write(timeStamp + "\n\n");
            //2. Nahasmen matrizeak
            fw.write("5-fCV: \n\n");
            fw.write(ev1.toMatrixString());
            fw.write("\n\nHold-out (%70): \n\n");
            fw.write(ev2.toMatrixString());
            //3. Programan sartutako parametroak
            fw.write("\nProgramaren argumentuak: \n\n");
            fw.write("args[0] datuen path (input): " + args[0] + "\n");
            fw.write("args[1] eredua gordetzeko path (output): " + args[1] + "\n");
            fw.write("args[2] kalitatearen estimazioa gordetzeko path (output): " + args[2] + "\n");
            fw.close();
        }
        catch(IOException e) {
            System.out.println("ERROREA: Emaitza fitxategiaren path-a berrikusi: " + args[1]);
        }
        
        ////2. PROGRAMA --> "evaluateModelOnce(AndRecordPredicition)"
        //modeloa dagoeneko entrenatuta dago --> fc.buildClassifier egitea EZ DA BEHARREZKOA

        
    }
}


package labo6;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class ToyStringExampleSetInputFormat {

	public static void main(String[] args) throws Exception {
		////ARGUMENTUAK: 

		// args[0] = train multzoa (input)						-->	/home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/Datuak/toyStringExample_train_RAW.arff
		// args[1] = test multzoa (input)						-->	/home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/Datuak/toyStringExample_test_RAW.arff
		// args[2] = trainBoW [sparse] multzoa (output)  		--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/toyStringExample_train_BoW_sparse.arff
		// args[3] = trainBoW [non-sparse] multzoa (output) 	--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/toyStringExample_train_BoW_non_sparse.arff
		// args[4] = s2w filtrotik lortutako hiztegia (output)  --> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/dictionary.txt
		// args[5] = testBoW multzoa [sparse] (output)  		--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/toyStringExample_test_BoW_sparse.arff
		// args[6] = testBoW multzoa [non-sparse] (output)  	--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/toyStringExample_test_BoW_non_sparse.arff		

		
		////DATUAK KARGATU (split-a dagoeneko sortuta dago) [TRAIN ETA TEST]
        DataSource sourceTrain = new DataSource(args[0]);
        Instances trainRaw = sourceTrain.getDataSet();
        if(trainRaw.classIndex() == -1)
            trainRaw.setClassIndex(trainRaw.numAttributes() - 1);
        
        DataSource sourceTest = new DataSource(args[1]);
        Instances testRaw = sourceTest.getDataSet();
        if(testRaw.classIndex() == -1)
            testRaw.setClassIndex(testRaw.numAttributes() - 1);
        
        ////ENTRENAMENDU SORTA BOW ERAN ADIERAZI (sparse)
        StringToWordVector s2w_sparse = new StringToWordVector();
        s2w_sparse.setWordsToKeep(Integer.MAX_VALUE); //gordeko den hitz kopuru MAX
        s2w_sparse.setOutputWordCounts(false); //FALSE = BOOLEAN; true = maiztasuna
        s2w_sparse.setLowerCaseTokens(true); //hitz guztiak hizki xehez
        s2w_sparse.setInputFormat(trainRaw);
        Instances trainBoW_sparse = Filter.useFilter(trainRaw, s2w_sparse);
        
        ////ENTRENAMENDU SORTA BOW ERAN ADIERAZI (non-sparse)
	    SparseToNonSparse sTOns = new SparseToNonSparse();
	    sTOns.setInputFormat(trainBoW_sparse);
	    Instances trainBoW_nonSparse = Filter.useFilter(trainBoW_sparse, sTOns);
        
        ////TEST SORTA LORTU (BOW [SPARSE])
	    Remove rmFilter = new Remove();
	    rmFilter.setInvertSelection(false); //false = hautatutako zutabeak EZABATU; true = hautatutako zutabeak MANTENDU + gainontzekoak ezabatu
	    rmFilter.setInputFormat(trainBoW_sparse);
        Instances testBoW_sparse = Filter.useFilter(testRaw, rmFilter); 
  
        ////TEST SORTA NON-SPARSE ERAN ESKURATU
	    sTOns.setInputFormat(testBoW_sparse);
	    Instances testBoW_nonSparse = Filter.useFilter(testBoW_sparse, sTOns);
        
        
        ////DATUAK ESPORTATU (ONDO DAUDELA EGIAZTATZEKO)
        ArffSaver saveTrainBowSparse = new ArffSaver();
        saveTrainBowSparse.setInstances(trainBoW_sparse);
        saveTrainBowSparse.setFile(new File(args[2]));
        saveTrainBowSparse.writeBatch();
        
        ArffSaver saveTrainBowNonSparse = new ArffSaver();
        saveTrainBowNonSparse.setInstances(trainBoW_nonSparse);
        saveTrainBowNonSparse.setFile(new File(args[3]));
        saveTrainBowNonSparse.writeBatch();
        
        ArffSaver saveTestBowSparse = new ArffSaver();
        saveTestBowSparse.setInstances(testBoW_sparse);
        saveTestBowSparse.setFile(new File(args[5]));
        saveTestBowSparse.writeBatch();
        
        ArffSaver saveTestBowNonSparse = new ArffSaver();
        saveTestBowNonSparse.setInstances(testBoW_nonSparse);
        saveTestBowNonSparse.setFile(new File(args[6]));
        saveTestBowNonSparse.writeBatch();
	}
}

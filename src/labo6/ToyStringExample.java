package labo6;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class ToyStringExample {

	public static void main(String[] args) throws Exception {
		////ARGUMENTUAK: 

		// args[0] = train multzoa (input)						-->	/home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/Datuak/toyStringExample_train_RAW.arff
		// args[1] = test multzoa (input)						-->	/home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/Datuak/toyStringExample_test_RAW.arff
		// args[2] = trainBoW [sparse] multzoa (output)  		--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/toyStringExample_train_BoW_sparse.arff
		// args[3] = trainBoW [non-sparse] multzoa (output) 	--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/toyStringExample_train_BoW_non_sparse.arff
		// args[4] = s2w filtrotik lortutako hiztegia (output)  --> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/dictionary.txt
		// args[5] = testBoW multzoa (output)  					--> /home/jfu/Escritorio/EHES/Labo6/1_ToyStringExample/Datuak/toyStringExample_test_BoW.arff		
		
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
        s2w_sparse.setOutputWordCounts(false); //FALSE = SPARSE [0 DIRENAK SOILIK AGERTU]; true = non_sparse
        s2w_sparse.setLowerCaseTokens(true); //hitz guztiak hizki xehez
        s2w_sparse.setSaveDictionaryInBinaryForm(false); //hiztegia testu leuneko fitxategi gisa gorde
        s2w_sparse.setDictionaryFileToSaveTo(new File(args[4]));
        s2w_sparse.setInputFormat(trainRaw);
        Instances trainBoW_sparse = Filter.useFilter(trainRaw, s2w_sparse);
        
        ////ENTRENAMENDU SORTA BOW ERAN ADIERAZI (non-sparse)
	    SparseToNonSparse sTOns = new SparseToNonSparse();
	    sTOns.setInputFormat(trainBoW_sparse);
	    Instances trainBoW_nonSparse = Filter.useFilter(trainBoW_sparse, sTOns);
        
        ////S2W-TIK "DICTIONARY" ESKURATU ETA FDS2W FILTROA OSATU
        FixedDictionaryStringToWordVector fds2w = new FixedDictionaryStringToWordVector();
        fds2w.setDictionaryFile(s2w_sparse.getDictionaryFileToSaveTo());
        fds2w.setInputFormat(trainBoW_sparse);
        Instances testBow = Filter.useFilter(testRaw, fds2w); //instantziarik sortzen da. ZERGATIK !!??
        System.out.println(testRaw);
        
        ////DATUAK ESPORTATU (ONDO DAUDELA EGIAZTATZEKO)
        ArffSaver saveTrainBowSparse = new ArffSaver();
        saveTrainBowSparse.setInstances(trainBoW_sparse);
        saveTrainBowSparse.setFile(new File(args[2]));
        saveTrainBowSparse.writeBatch();
        
        ArffSaver saveTrainBowNonSparse = new ArffSaver();
        saveTrainBowNonSparse.setInstances(trainBoW_nonSparse);
        saveTrainBowNonSparse.setFile(new File(args[3]));
        saveTrainBowNonSparse.writeBatch();
        
        ArffSaver saveTest = new ArffSaver();
        saveTest.setInstances(testBow);
        saveTest.setFile(new File(args[5]));
        saveTest.writeBatch();
	}
}

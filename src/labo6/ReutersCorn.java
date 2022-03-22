package labo6;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class ReutersCorn {
	    
    public static void main(String[] args) throws Exception {
        ////DATUAK KARGATU (split-a dagoeneko sortuta dago)
        DataSource sourceTrain = new DataSource("/home/lsi/Descargas/ReutersCorn-train.arff");
        Instances trainRaw = sourceTrain.getDataSet();
        if(trainRaw.classIndex() == -1)
            trainRaw.setClassIndex(trainRaw.numAttributes() - 1);
        
        DataSource sourceTest = new DataSource("/home/lsi/Descargas/ReutersCorn-test.arff");
        Instances devRaw = sourceTest.getDataSet();
        if(devRaw.classIndex() == -1)
            devRaw.setClassIndex(devRaw.numAttributes() - 1);
        
        ////ENTRENAMENDU SORTA BOW ERAN ADIERAZI
        StringToWordVector s2w = new StringToWordVector();
        s2w.setInputFormat(trainRaw);
        Instances trainBoW = Filter.useFilter(trainRaw, s2w);
        
        FixedDictionaryStringToWordVector fds2w = new FixedDictionaryStringToWordVector();
        s2w.getDictionaryFileToSaveTo(new File("/home/lsi/Escritorio/dict.arff"));
        fds2w.setDictionaryFile();
        
        ////DATUAK ONDO DAUDELA EGIAZTATZEKO
        ArffSaver saveTrain = new ArffSaver();
        saveTrain.setInstances(trainBoW);
        saveTrain.setFile(new File("/home/lsi/Escritorio/trainBow.arff"));
        saveTrain.writeBatch();
        

	
}

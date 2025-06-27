
package weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;


public class Weka {

    
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        BufferedReader buffReader = new BufferedReader(new FileReader("C:\\Program Files\\Weka-3-8-6\\data\\iris.arff"));
        
        Instances train = new Instances(buffReader);
        train.setClassIndex(train.numAttributes()-1);
        
        buffReader.close();
        
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);
        
        Evaluation eval =new Evaluation(train);
        
        eval.crossValidateModel(nb, train, 10, new Random(1));
        
        System.out.println(eval.toSummaryString("\nResults:\n--------\n", true));
        
        System.out.println(eval.fMeasure(1)+" "+ eval.precision(1));
    }
    
}

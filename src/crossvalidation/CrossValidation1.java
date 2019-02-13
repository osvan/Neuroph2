package crossvalidation;

import java.util.concurrent.ExecutionException;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.*;

import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

/**
 *
 * @author darku
 */
public class CrossValidation1 {

    
     

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        //Define input layer
        Layer inputLayer = new Layer();
        inputLayer.addNeuron(new Neuron());
        inputLayer.addNeuron(new Neuron());

        //Implement hidden layer 1
        Layer hiddenLayerOne = new Layer();
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());
        hiddenLayerOne.addNeuron(new Neuron());

        //Implement hidden layer 2
        Layer hiddenLayerTwo = new Layer();
        hiddenLayerTwo.addNeuron(new Neuron());
        hiddenLayerTwo.addNeuron(new Neuron());
        hiddenLayerTwo.addNeuron(new Neuron());
        hiddenLayerTwo.addNeuron(new Neuron());

        //Output
        Layer outputLayer = new Layer();
        outputLayer.addNeuron(new Neuron());

        //Neural Network
        NeuralNetwork ann = new NeuralNetwork();
        ann.addLayer(0, inputLayer);
        ann.addLayer(1, hiddenLayerOne);
        ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
        ann.addLayer(2, hiddenLayerTwo);
        ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
        ann.addLayer(3, outputLayer);
        ConnectionFactory.fullConnect(ann.getLayerAt(2), ann.getLayerAt(3));
        ConnectionFactory.fullConnect(ann.getLayerAt(0),
                ann.getLayerAt(ann.getLayersCount() - 1), false);
        ann.setInputNeurons(inputLayer.getNeurons());
        ann.setOutputNeurons(outputLayer.getNeurons());
        
       

        System.out.println("Neuronas en capa 0: " + ann.getLayerAt(0).getNeuronsCount());
        System.out.println("Neuronas en capa 1: " + ann.getLayerAt(1).getNeuronsCount());
        System.out.println("Neuronas en capa 2: " + ann.getLayerAt(2).getNeuronsCount());
        System.out.println("Neuronas en capa 3: " + ann.getLayerAt(3).getNeuronsCount());

        //Training
//        int inputSize = 2;
//        int outputSize = 1;
//        DataSet ds = new DataSet(inputSize, outputSize);
//        
//        
//        //add elements to dataset
//        DataSetRow rOne
//                = new DataSetRow(new double[]{0, 1}, new double[]{1});
//        ds.addRow(rOne);
//        DataSetRow rTwo
//                = new DataSetRow(new double[]{1, 1}, new double[]{0});
//        ds.addRow(rTwo);
//        DataSetRow rThree
//                = new DataSetRow(new double[]{0, 0}, new double[]{0});
//        ds.addRow(rThree);
//        DataSetRow rFour
//                = new DataSetRow(new double[]{1, 0}, new double[]{1});
//        ds.addRow(rFour);
        //importar dataset
        String inputFileName = "C:\\Users\\darku\\Documents\\Datasets\\test.csv";
        DataSet dataSet = DataSet.createFromFile(inputFileName, 2, 1, ",", false);

        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(1000);
        ann.learn(dataSet, backPropagation);
        
        CrossValidation crossval = new CrossValidation(ann, dataSet, 5);
                                                                                  
        crossval.run();
        CrossValidationResult results = crossval.getResult();
        results.printResult();  
        

        //Testing 0 1
        ann.setInput(0, 1);
        ann.calculate();
        double[] networkOutputOne = ann.getOutput();
        print("0, 1", networkOutputOne[0], 1.0);

        //Testing 1 1
        ann.setInput(1, 1);
        ann.calculate();
        print("1, 1", ann.getOutput()[0], 0.0);

    }

    public static void print(String input, double output, double actual) {
        System.out.println("Testing: " + input + " Expected: " + actual + " Result: " + output);
    }

}

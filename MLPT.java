import java.io.*;
import java.util.*;
import java.util.Scanner;
import java.util.StringTokenizer;

public class MLPT {
	private int numberOfLayers ;
	private int numberOfInputs ;
	private int numberOfOutputs ;
	private int numberOfPatterns;
	private int numberOfTestPatterns;
	private int numberOfHiddenNeurons ;
	private Layer[] layers;
	private double[] currentInputs;
	private double[] currentOutputs;
	private double[] currentHiddenOutputs;
	private double[][] networkOutputs;
	private double[][] networkHiddenOutputs;
	
	private double[][] patterns;
	private double[][] hiddenPatterns;
	private double[][] outPatterns;
	private double[][] testPatterns;
	

	private double[][] hiddenWeights;
	private double[][] outWeights;
	private Normalizer nml;
	private String trainFile;
	private String testFile;
	
	
	public MLPT(int nl, int ni,int nh, int no, int np) {
		numberOfLayers= nl;
		numberOfInputs = ni;
		numberOfOutputs = no ;
		numberOfHiddenNeurons = nh ;
		numberOfPatterns=np;
		layers = new Layer[2];
		layers[0] = new Layer(2,2);
		layers[1] = new Layer(1,2);
		patterns = new double[numberOfPatterns][numberOfInputs+numberOfOutputs];
		hiddenPatterns = new double[numberOfPatterns][numberOfInputs+numberOfHiddenNeurons];
		outPatterns = new double[numberOfPatterns][numberOfHiddenNeurons+numberOfOutputs];

	}
	public MLPT() {
		this(2,2,2,1,4);
		double[][] pats = { {-1, -1, -1, -1}, {-1, 1, 1, -1}, {1, -1, 1, -1}, {1, 1, 1, 1} };
		patterns=pats;
	}
	public MLPT(String fname) throws IOException {
		this();
		nml = new Normalizer(fname);
		nml.normalize();

	}

	public void readMLPT() throws IOException {
		//Scanner sc = new Scanner(new File("mlpdata1.txt"));
		System.out.println("Reading data of MLP from "+trainFile);
		Scanner sc = new Scanner(new File(trainFile));
		numberOfLayers = 2;
		numberOfInputs = sc.nextInt();
		numberOfHiddenNeurons = 20;
		numberOfOutputs = sc.nextInt();
		numberOfPatterns = sc.nextInt();
		layers = new Layer[numberOfLayers];
		layers[0] = new Layer(numberOfHiddenNeurons,numberOfInputs,numberOfPatterns,"Hidden");
		layers[1] = new Layer(numberOfOutputs,numberOfHiddenNeurons,numberOfPatterns,"Output");
		patterns = new double[numberOfPatterns][numberOfInputs+numberOfOutputs];
		for ( int p = 0; p < numberOfPatterns; p++) {
			for ( int i = 0; i < numberOfInputs+numberOfOutputs ; i ++ ) {
				patterns[p][i] = sc.nextDouble();
			}
		}
		hiddenPatterns = new double[numberOfPatterns][numberOfInputs+numberOfHiddenNeurons];
		outPatterns = new double[numberOfPatterns][numberOfHiddenNeurons+numberOfOutputs];
		for ( int p = 0; p < numberOfPatterns; p++) {
			for ( int i = 0; i < numberOfInputs; i ++ ) {
				hiddenPatterns[p][i] = patterns[p][i];
			}
			//for ( int i = numberOfHiddenNeurons; i < numberOfHiddenNeurons+numberOfOutputs; i ++ ) {
			for ( int i = numberOfInputs; i < numberOfInputs+numberOfOutputs; i ++ ) {
				outPatterns[p][numberOfHiddenNeurons + i -numberOfInputs] = patterns[p][i];
			}
		}
		sc.close();
		readTestPatterns();
		
		layers[0].assignPatterns(hiddenPatterns);
		layers[1].assignPatterns(outPatterns);
		//layers[0].printPatterns();
		//layers[1].printPatterns();

		hiddenWeights = new double[numberOfHiddenNeurons][numberOfInputs+1];
		outWeights = new double[numberOfOutputs][numberOfHiddenNeurons+1];
		randomizeWeights(0.5);
		currentInputs = new double[numberOfInputs+1];
		currentHiddenOutputs = new double[numberOfHiddenNeurons+1];
		currentOutputs= new double[numberOfOutputs];
		networkOutputs =  new double[numberOfPatterns][numberOfOutputs];
		networkHiddenOutputs =  new double[numberOfPatterns][numberOfHiddenNeurons+1];
	}
	
	public int readTestPatterns() throws IOException {
		System.out.println("Reading data of MLP from "+testFile);
		Scanner sc = new Scanner(new File(testFile));
		int numberOfTestLayers = 2;
		int numberOfTestInputs = sc.nextInt();
		int numberOfTestHiddenNeurons = 100;
		int numberOfTestOutputs = sc.nextInt();
		numberOfTestPatterns = sc.nextInt();
		testPatterns = new double[numberOfTestPatterns][numberOfInputs+numberOfOutputs];
		if(numberOfTestInputs != numberOfInputs) {
			System.out.println("Number of test and train inputs dont match : exiting tests ");
			return -1;
		}			
		if(numberOfTestOutputs != numberOfOutputs) {
			System.out.println("Number of test and train outputs dont match : exiting tests ");
			return -2;
		}			
		testPatterns = new double[numberOfTestPatterns][numberOfInputs+numberOfOutputs];
		for ( int p = 0; p < numberOfTestPatterns; p++) {
			for ( int i = 0; i < numberOfInputs+numberOfOutputs ; i ++ ) {
				testPatterns[p][i] = sc.nextDouble();
			}
		}
		sc.close();
		return 1;	
	}	
	
	
	public void setTrainFile( String fileName ) {
		trainFile = fileName;
	}

	public void setTestFile( String fileName ) {
		testFile = fileName;
	}
	
	public void randomizeWeights ( double range) { // weights to be randomized in (-range,+range)
		layers[0].randomizeWeights (range);
		layers[1].randomizeWeights (range);
	}
	
	public void saveWeights ( String fileName ) {
		try {
			FileWriter fw = new FileWriter( fileName);
			PrintWriter pw = new PrintWriter( fw);
			layers[0].saveWeights (pw);
			layers[1].saveWeights (pw);
			pw.close();
			fw.close();
		}
		catch (IOException ioe ) {
			System.out.println(" Error saving weights in file "+fileName );
		}
	}
	
	public void readWeights ( String fileName ) {
		try {
			Scanner sc = new Scanner(new File(fileName));
			//System.out.println("Reading weights of MLP from file "+fileName);
			for ( int n = 0; n < numberOfHiddenNeurons ; n++ ) {
				for ( int i = 0; i <= numberOfInputs ; i ++ ) {
					hiddenWeights[n][i] = sc.nextDouble() ;
				}
			}
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				for ( int i = 0; i <= numberOfHiddenNeurons ; i ++ ) {
					outWeights[n][i] = sc.nextDouble() ;
				}
			}
			layers[0].assignWeights(hiddenWeights);
			layers[1].assignWeights(outWeights);
		}
		catch (IOException ioe ) {
			System.out.println(" Error reading weights from file "+fileName );
		}
	}		
			
	 
	public void printPatterns() { 
		System.out.println("Patterns of the MLP are : ");
		for ( int p = 0; p < numberOfPatterns; p++) {
			System.out.print(" For pattern: "+(p+1)+" inputs are : ");
			for ( int i = 0; i < numberOfInputs ; i ++ ) {
				System.out.print(patterns[p][i] +", ");
			}
			System.out.print(" and outputs ");
			for ( int i = numberOfInputs; i < numberOfInputs+numberOfOutputs ; i ++ ) {
				System.out.print(patterns[p][i] +", ");
			}
			System.out.println();
		}
	}

	public void printWeights() { 
		layers[0].printWeights();
		layers[1].printWeights();
	}

	public void  feedForward (int pat ) {
		currentInputs[0] = -1.0 ;
		for ( int i = 0; i < numberOfInputs ; i ++ ) {
			currentInputs[i+1] = patterns[pat][i];
		}
		double[] hiddenOutputs = new double[numberOfHiddenNeurons] ;
		hiddenOutputs = layers[0].feedForward(currentInputs,pat);
		currentHiddenOutputs[0] = -1.0;
		networkHiddenOutputs[pat][0] =-1.0 ;
		for ( int n = 0; n < numberOfHiddenNeurons ; n++ ) {
			currentHiddenOutputs[n+1] = hiddenOutputs[n] ;
			networkHiddenOutputs[pat][n+1] =hiddenOutputs[n] ;	
		}
		currentOutputs = layers[1].feedForward(currentHiddenOutputs,pat );
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			networkOutputs[pat][n] = currentOutputs[n];
		}
	}

	public void  feedForward () {
		for ( int p = 0; p < numberOfPatterns; p++) {
			feedForward ( p );
		}
		//printNetworkOutputs ();
	}


	public double[] errorBackPropagate (int pat) {
		currentInputs[0] = -1.0 ;
		for ( int i = 0; i < numberOfInputs ; i ++ ) {
			currentInputs[i+1] = patterns[pat][i];
		}		
		double[] outErrors = new double[numberOfOutputs];
		double[] hiddenErrors = new double[numberOfHiddenNeurons+1];
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			outErrors[n] = patterns[pat][numberOfInputs+n] - networkOutputs[pat][n];
		}
		for ( int n = 0; n <= numberOfHiddenNeurons ; n++ ) {
			double hidError = 0.0;
			for ( int on = 0; on < numberOfOutputs ; on++ ) {
				hidError += 0.5 * outErrors[on]*(1.0 - networkOutputs[pat][on] * networkOutputs[pat][on])*outWeights[on][n];
			}
			hiddenErrors[n] = hidError;
		}

		outWeights = layers[1].errorBackPropagate(outErrors,networkHiddenOutputs,pat);
		hiddenWeights = layers[0].errorBackPropagate(hiddenErrors,currentInputs);
		return outErrors;
	}

	public double errorBackPropagate () {
		double[] sumSquareErrors = new double[numberOfOutputs];
		double[] patternErrors = new double[numberOfOutputs];	
		for ( int p = 0; p < numberOfPatterns; p++) {
			patternErrors = errorBackPropagate(p);
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				sumSquareErrors[n] += patternErrors[n] * patternErrors[n] ;
			}
		}
		double totalrmse = 0.0;
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			sumSquareErrors[n] /= numberOfPatterns ;
			sumSquareErrors[n] = Math.sqrt(sumSquareErrors[n]);
			totalrmse += sumSquareErrors[n] ;			
		}
		return totalrmse ;
	}

	public double[] train ( int pat ) {
		feedForward ( pat );
		double[] patternErrors = new double[numberOfOutputs];		
		patternErrors = errorBackPropagate(pat);
		return patternErrors ;
	}

	public double train (  ) {
		double[] sumSquareErrors = new double[numberOfOutputs];
		double[] patternErrors = new double[numberOfOutputs];	
			
		for ( int p = 0; p < numberOfPatterns; p++) {
			patternErrors = train(p);
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				sumSquareErrors[n] += patternErrors[n] * patternErrors[n] ;
			}
			
		}
		double totalrmse = 0.0;
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			sumSquareErrors[n] /= numberOfPatterns ;
			sumSquareErrors[n] = Math.sqrt(sumSquareErrors[n]);
			totalrmse += sumSquareErrors[n] ;			
		}
		//printNetworkOutputs ();
		return round(totalrmse,6) ;
	}

	public double[] testPattern ( double[] patternData) {
		currentInputs[0] = -1.0 ;
		for ( int i = 0; i < numberOfInputs ; i ++ ) {
			currentInputs[i+1] = patternData[i];
		}	
		double[] hiddenOutputs = new double[numberOfHiddenNeurons] ;
		hiddenOutputs = layers[0].feedForward(currentInputs);
		currentHiddenOutputs[0] = -1.0;
		for ( int n = 0; n < numberOfHiddenNeurons ; n++ ) {
			currentHiddenOutputs[n+1] = hiddenOutputs[n] ;
		}
		double[] outErrors;
		currentOutputs = layers[1].feedForward(currentHiddenOutputs );				
		if ( patternData.length > numberOfInputs ) {
			outErrors = new double[numberOfOutputs];
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				outErrors[n] = patternData[numberOfInputs+n] - currentOutputs[n];
			}
		}	
		else {
			outErrors = new double[1];
			outErrors[0] = 0.0;
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				System.out.print(currentOutputs[n]+"   :   ");
			}
			System.out.println();
			//layers[1].printOutputs();			
		}			
		return outErrors;
	}
	
	public double[] testPattern ( int tpn) {
		currentInputs[0] = -1.0 ;
		for ( int i = 0; i < numberOfInputs ; i ++ ) {
			currentInputs[i+1] = testPatterns[tpn][i];
		}	
		double[] hiddenOutputs = new double[numberOfHiddenNeurons] ;
		hiddenOutputs = layers[0].feedForward(currentInputs);
		currentHiddenOutputs[0] = -1.0;
		for ( int n = 0; n < numberOfHiddenNeurons ; n++ ) {
			currentHiddenOutputs[n+1] = hiddenOutputs[n] ;
		}
		currentOutputs = layers[1].feedForward(currentHiddenOutputs );				
		double outError = 0.0;
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			double neuronError =  testPatterns[tpn][numberOfInputs+n] - currentOutputs[n];
			outError += neuronError * neuronError;
		}
		outError = Math.sqrt(outError);
		outError /= numberOfOutputs; 
		System.out.println();	
		System.out.print(" For Pattern number "+tpn+" with inputs : ");	
		for ( int i = 0; i < numberOfInputs ; i ++ ) {
			System.out.print(currentInputs[i]+ " : " );
		}	
		System.out.println();	
		System.out.print(" The computed and desired output paired values : ");	
		for ( int i = 0; i < numberOfOutputs ; i ++ ) {
			System.out.print(currentOutputs[i]+" ~ "+testPatterns[tpn][numberOfInputs+i]+ " : " );
		}			
		System.out.println();
		System.out.println(" RMS error = "+outError);
		double[] outErrors = new double[1];
		outErrors[0] = outError;
		return outErrors;
	}
	public void test() {
		System.out.println(" Type the test pattern number in the range ( 1 , "+numberOfTestPatterns+
			") or type "+numberOfInputs+" input values to test or -1 to exit:" );
		Scanner sc = new Scanner ( System.in);
		String input = sc.nextLine();
		StringTokenizer st = new StringTokenizer(input);
		int nt = st.countTokens();	
		String tkn = st.nextToken();
		double val = Double.parseDouble(tkn);
		double[] inps = new double[numberOfInputs];
		if ( nt == 1 ) {
			if( numberOfInputs == 1 ) {
				inps[0] = val;
			}
			else if ( val < 0.0 ) {
				return ;
			}
			else {
				int ptn = (int)val;
				if ( ptn > numberOfTestPatterns ) {
					System.out.println(" There are no more than "+numberOfTestPatterns+" test patterns read. ");
				}
				else {
					testPattern (ptn);
				}
			}
		}
		else if ( nt == numberOfInputs) {
			inps[0] = val;
			for ( int i = 1; i < numberOfInputs; i ++ ) {
				tkn = st.nextToken();
				val = Double.parseDouble(tkn);				
				inps[i] =  val;
			}
			testPattern (inps);
		}
		else { 
			System.out.println(" Sufficient inputs not given  ");
		}	
	}
	
	public double testAll (  ) {
		double[] sumSquareErrors = new double[numberOfOutputs]; 
		double[] patternErrors = new double[numberOfOutputs];			
		for ( int p = 0; p < numberOfTestPatterns; p++) {
			double[] testPatternData = new double[numberOfInputs + numberOfOutputs];
			for ( int i = 0; i < numberOfInputs+numberOfOutputs ; i ++ ) {
				testPatternData[i] = testPatterns[p][i];
			}			
			patternErrors = testPattern(testPatternData);
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				sumSquareErrors[n] += patternErrors[n] * patternErrors[n] ;
			}			
		}
		double totalrmse = 0.0;
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			sumSquareErrors[n] /= numberOfPatterns ;
			sumSquareErrors[n] = Math.sqrt(sumSquareErrors[n]);
			totalrmse += sumSquareErrors[n] ;			
		}
		return round(totalrmse,6) ;
	}



	public void printOutputs () {
		//System.out.println(" Hidden Layer Outputs ");
		layers[0].printOutputs();
		//System.out.println(" Output Layer Outputs ");
		layers[1].printOutputs();
	}
	
	public double round ( double number, int places ) {
		double power = 10.0;
		for( int i = 1 ; i < places ; i ++ ) power *= 10.0;
		double rounded = (long) (number * power );
		rounded /= power ;
		return rounded ;
	}

	public void printNetworkOutputs () {
		double[] rmse = new double[numberOfOutputs] ;
		for ( int p = 0; p < numberOfPatterns; p++) {
			System.out.print(" p"+(p+1)+" : ");
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				double desired = patterns[p][numberOfInputs+n] ;
				double error = desired - networkOutputs[p][n] ;
				rmse[n] += error * error ;
				System.out.print( round (networkOutputs[p][n],4)+"("+round(desired,4)+ ")" );
			}
		}
		System.out.println();
		System.out.print( " RMS Errors : ");		
		for ( int n = 0; n < numberOfOutputs ; n++ ) {
			rmse[n] = Math.sqrt(rmse[n] / numberOfPatterns );
			System.out.print(rmse[n]+" ; ");
		}			
		System.out.println();
	}
	
	public void testTESTSET( int cycles) throws IOException {
		PrintWriter pr = new PrintWriter( new FileWriter("TESTSETtrybundy"+cycles+"cycles.txt"));	
		int numberOfTestPatterns = 30;
		for (int tpn = 0; tpn < numberOfTestPatterns;tpn++) {
			currentInputs[0] = -1.0 ;
			for ( int i = 0; i < numberOfInputs ; i ++ ) {
				currentInputs[i+1] = testPatterns[tpn][i];
			}	
			double[] hiddenOutputs = new double[numberOfHiddenNeurons] ;
			hiddenOutputs = layers[0].feedForward(currentInputs);
			for (int i=0 ; i<numberOfHiddenNeurons; i++){
			System.out.println("The Problematic hiddenOUTPUTS "+hiddenOutputs[i]);
			}				
			currentHiddenOutputs[0] = -1.0;
			for ( int n = 0; n < numberOfHiddenNeurons ; n++ ) {
				currentHiddenOutputs[n+1] = hiddenOutputs[n] ;
				System.out.println("The Problematic currentHIDDENOUTPUTS "+currentHiddenOutputs[n+1]);				
			}
			currentOutputs = layers[1].feedForward(currentHiddenOutputs );
			System.out.println("The Problematic currentOUTPUTS "+currentOutputs);	
			System.out.println("no of inputs"+numberOfInputs);			
			double outError = 0.0;
			for ( int n = 0; n < numberOfOutputs ; n++ ) {
				double neuronError =  testPatterns[tpn][numberOfInputs+n] - currentOutputs[n];
				outError += neuronError * neuronError;
			}
			outError = Math.sqrt(outError);
			outError /= numberOfOutputs; 
			System.out.println();	
			
			
			double[] outErrors = new double[1];
			outErrors[0] = outError;
		
			double[] inputData = new double  [numberOfInputs+numberOfOutputs];
			double[] outputData = new double  [numberOfInputs+numberOfOutputs];
			for (int i=0 ; i<numberOfInputs+numberOfOutputs; i++) {
				if (i<numberOfInputs) {
					inputData [i] = testPatterns [tpn][i];
					System.out.println("inputdata from pattern (0 to 16)"+inputData[i]);
				}
				else {
					inputData [i] = currentOutputs[i-numberOfInputs];
					System.out.println("inputdata for 17 only currentoutput"+inputData[i]);
				}
			}
	
			outputData = nml.remap(inputData);
			
		
			double[] desiredOutput = new double [numberOfOutputs];
			for (int k=0; k < numberOfOutputs ; k++) {
				desiredOutput [k] = testPatterns [tpn][numberOfInputs+k];
			}
			double[] rmpdesiredOutput = new double [numberOfOutputs];
			rmpdesiredOutput = nml.remap1(desiredOutput);
			
			pr.println();
			pr.println(" For Pattern number "+(tpn+1)+" with inputs as different parameters: ");	
			for ( int i = 0; i < numberOfInputs ; i ++ ) {
				pr.print(outputData[i]+"  " );
			}
			pr.println();
			pr.print("The computed Output percentage of wastage is : ");
			for (int i=numberOfInputs; i<numberOfInputs+numberOfOutputs; i++) {
				pr.print(outputData[i]+"  ");
			}
			pr.println();
			pr.print("The desired Output percentage of wastage is : " );
			for (int i=0; i<numberOfOutputs; i++) {
				pr.print(rmpdesiredOutput[i]+"  ");
			}
			pr.println();
			
			double realErr = 0;
			for (int i=0; i<numberOfOutputs; i++) {
				double err = Math.abs( rmpdesiredOutput[i] - outputData[i+numberOfInputs]);
				realErr = realErr + (err*err);
			}
			pr.print("The RMS Error : "+realErr); 
			
		}
		pr.close();
		/*double [] abc = new double [1];
		abc [0] = 0;
		return abc;	*/
	}
			
	public static void main (String[] args ) throws IOException {
		MLPT m1 = new MLPT("data.txt");	
		int cycles = 800000;
		String trainFile = "trainnormdata.txt";
		String testFile = "testnormdata.txt";
		
		if ( args.length > 2 )
			testFile = args[2] ;
		if ( args.length > 1 )
			trainFile = args[1] ;			
		if ( args.length >0 ) {
			cycles = Integer.parseInt(args[0]);
		}
		m1.setTrainFile(trainFile);
		m1.setTestFile(testFile);
		
		m1.readMLPT();
		//m1.printPatterns();
		m1.randomizeWeights (0.5);
		//m1.printWeights();		
		FileWriter fw = new FileWriter( "trainbundy" +cycles +"cycles.txt");
		PrintWriter pw = new PrintWriter( fw);
		int k =1;	
		pw.println("during training cycle "+"	,"+" train error "+",  "+"test error " +",   "+ args[0]);	
		for ( int c = 0 ; c < cycles ;c++  ) {
			//System.out.println("#############Training Cycle Number "+(c+1));
			double rmse = m1.train();
			double testSetError = m1.testAll();
			
			if (c == 10*k ) {
			pw.println((c+1)+"   ,	"+rmse+",   "+testSetError);
			//pw.println(" during training cycle "+(c+1)+" train error = "+rmse+" test error = "+testSetError);
			k++;
			}
			
		}
		
		
		pw.close();
		fw.close();
		//System.out.println(" Reading and testing the final weights ");
		m1.saveWeights("weightsFilebundy"+cycles);
		m1.readWeights("weightsFilebundy"+cycles);				
		m1.feedForward();
		int choice = 0;
		m1.test();
		m1.testTESTSET(cycles);
		//m1.printNetworkOutputs();
	}
}




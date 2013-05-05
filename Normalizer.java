/*   This program is used to normalize the given data as to employ MLP, we require the input within
the range [-1,1] (because it is the domain of the activation function used in Neuron.java).
It contains mainly two functions called normalize () and remap(), both use linear map (a line passing 
through two points given by (x-x1)/(x2-x1)=(y-y1)/(y2-y1)=l ) to serve the purpose.
In normalize(), the map L : R to [-1,1] given by the above definition; where for a value x in R (i.e. data)
we have to find corresponding point y in [-1,1]. x1,x2,y1,y2 are min and max values for both x and y respectively.
This program also have the capability to divide the normalised patterns in to training and testing data set through the
method called dividePatterns(). So at the end of the execution of this program we will have 2 files, one with the 
normalised data to be used as training data set and the other one as normalised data to be used as testing data set.
*/




import java.io.*;
import java.util.Scanner;
import java.util.Vector;
public class Normalizer {
	//datatypes to be used should be defined initially.
	private double [][] bundyData;
	private double [][] normalData;
	private double [][] remapData;
	private int numberofInputs;
	private int numberofOutputs;
	private int numberofPatterns;
	private int numberOfTestPatterns;
	private int numberOfTrainPatterns;	
	private double inMin, inMax, outMin, outMax;
	private double [][] inputs;
	private double [][] outputs;
	private double [][] normInputs;
	private double [][] normOutputs;
	//private double[] baseCoordinates;   ---NOT required in BundyProject but was useful in Apollo Project.
	
	
	// constructor 
	public Normalizer (int ni, int no, int np) {
		numberofInputs = ni;
		numberofOutputs = no;
		numberofPatterns = np;
		bundyData = new double [numberofPatterns] [numberofInputs+numberofOutputs];
		inputs = new double [numberofPatterns] [numberofInputs];
		outputs = new double [numberofPatterns] [numberofOutputs];
		//baseCoordinates = new double[numberofInputs];
	}
	
	// consturctor which will also read the file which contains the data to be normalize.
	public Normalizer (String dataFile) throws IOException {
		System.out.println("Reading data from "+dataFile);
		Scanner sc = new Scanner(new File(dataFile));
		numberofInputs = sc.nextInt();
		numberofOutputs = sc.nextInt();
		numberofPatterns = sc.nextInt();
		bundyData = new double [numberofPatterns] [numberofInputs+numberofOutputs];
		inputs = new double [numberofPatterns] [numberofInputs];
		outputs = new double [numberofPatterns] [numberofOutputs];
		/*baseCoordinates = new double[numberofInputs];
		for (int i=0; i<numberofInputs; i++) {
			baseCoordinates[i] = sc.nextDouble();
		}*/
		
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofInputs+numberofOutputs ; j++) {
				bundyData[i][j] = sc.nextDouble();
				if ( j < numberofInputs)  
					inputs [i][j] = bundyData [i][j];
				else
					outputs [i][j-numberofInputs] = bundyData [i][j];
				
			}
		}		
	}

	// default constructor
	public Normalizer () {
		this (17,1,153);
	}
	
	
	// additional method to read the data if one don't want to use the constructor in specific cases.
	public void readData () throws IOException  {
		String dataFile = "data.txt";
		System.out.println("Reading data from "+dataFile);
		Scanner sc = new Scanner(new File(dataFile));
		numberofInputs = sc.nextInt();
		numberofOutputs = sc.nextInt();
		numberofPatterns = sc.nextInt();
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofInputs+numberofOutputs ; j++) {
				bundyData[i][j] = sc.nextDouble();
				if ( j < numberofInputs)  
					inputs [i][j] = bundyData [i][j];
				else
					outputs [i][j-numberofInputs] = bundyData [i][j];				
			}
		}
	}
	
	//method to divide the patterns in to testing and training data set
	public void dividePatterns(String patFileName, int testPercent ) throws IOException {
 	    	if ( testPercent < 10 || testPercent > 50 ) {
 	    		testPercent = 20 ;
 	    	}
 	    	String testFileName = "test"+patFileName;
 	    	String trainFileName = "train"+patFileName;  	    	
		System.out.println(testPercent+ " % of patterns to be moved to file  "+testFileName+ " remaining to "+trainFileName) ;
		FileReader fr = new FileReader( patFileName);
		BufferedReader br = new BufferedReader( fr);
		Vector <String> patterns = new Vector <String>() ;
		String patternString = br.readLine() ;
		while ( patternString != null ) {
			patterns.add( patternString ) ;
			patternString = br.readLine() ;
		}
		
		int numberOfPatterns = patterns.size();
		numberOfTestPatterns = numberOfPatterns * testPercent / 100;
		numberOfTrainPatterns = numberOfPatterns - numberOfTestPatterns;
		System.out.println(numberOfTestPatterns+ " patterns in file "+  testFileName);
		System.out.println(numberOfTrainPatterns+ " patterns in file "+  trainFileName);
		
		FileWriter fw = new FileWriter( testFileName);
		PrintWriter testw = new PrintWriter( fw);
		fw = new FileWriter( trainFileName);
		PrintWriter trainw = new PrintWriter( fw);		
		
		Vector<Integer> moved = new Vector<Integer>() ;
		int np = 0 ;
		String s = numberofInputs+" "+ numberofOutputs+" "+ numberOfTestPatterns;
		testw.println(s);
		while (np < numberOfTestPatterns ) {
			int rp =(int) Math.round( Math.random() * numberOfPatterns );
			if( rp >= numberOfPatterns) 
			continue;
			//System.out.println("*&^%***************** "+rp);
			if ( ! ( moved.contains ( rp ) )) {
				moved.add(rp) ;
				patternString = patterns.get(rp); 
				testw.println(patternString);
				np ++ ;
			}
		}

		s = numberofInputs+" "+ numberofOutputs+" "+ numberOfTrainPatterns;
		trainw.println(s);
		for( int p=0; p < numberOfPatterns; p ++ ) {
			if ( ! ( moved.contains ( p ) ) ) {
				patternString = patterns.get(p); 
				trainw.println(patternString);
			}
		}
		br.close();
		fr.close();
		testw.close();
		trainw.close();
		fw.close();
		
	}
	
	public double[] remap( double[] pat ) {					// for remapping of input and output both....

		double[] repat = new double[numberofInputs + numberofOutputs];
		for ( int i =0; i< numberofInputs; i++ ) {
			repat[i] =  (inMax - inMin) * ((pat[i]+1)/2) + inMin;
		}
		for ( int i =numberofInputs; i< numberofInputs+numberofOutputs; i++ ) {
			repat[i] =  (outMax - outMin) * ((pat[i]+1)/2) + outMin;
		}
		
		/*for ( int i =0; i< numberofInputs+numberofOutputs; i++ ) {
				System.out.println("the remapped data is "+ repat[i] );
		}*/
		return repat;
	}
	public double[] remap1( double[] pat ) {				// for remapping of output only..
		double[] repat = new double[numberofOutputs];
		for ( int i =0; i< numberofOutputs; i++ ) {
			repat[i] =  (outMax - outMin) * ((pat[i]+1)/2) + outMin;
		}
		
		/*for ( int i =0; i< numberofOutputs; i++ ) {
				System.out.println("the remapped data for output only is "+ repat[i] );
		}*/
		return repat;
	}
	
	
	public void minmax() {
		double inmin,inmax,outmin,outmax;
		inmin = inmax = inputs [0][0];
		outmin= outmax= outputs[0][0];
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofInputs; j++) {
				if (inmin > inputs [i][j])
					inmin = inputs [i][j];
				if (inmax < inputs [i][j])
					inmax = inputs [i][j];
			}
			
			for (int j=0; j<numberofOutputs; j++) {
				if (outmin > outputs [i][j])
					outmin = outputs [i][j];
				if (outmax < outputs [i][j])
					outmax = outputs [i][j];
			}
		}
		inMin = inmin;
		inMax = inmax;
		outMin = outmin;
		outMax = outmax;
				
	}
	
	public double[] getNormalizationParameters() {
		double [] npm = new double[4];
		npm[0]=inMin;
		npm[1]=inMax;
		npm[2]=outMin;
		npm[3]=outMax;
		return npm;
	}
	

	
	public void normalize() {
		minmax();
		double pmax = inMax ;
		double pmin = inMin ;
		double xmax = outMax ;
		double xmin = outMin ;
		
		normInputs = new double [numberofPatterns] [numberofInputs];
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofInputs; j++) {
				normInputs[i][j] = ((2*(inputs[i][j] - pmin)) / (pmax - pmin)) - 1.0;
			}
		}
		
		normOutputs = new double [numberofPatterns] [numberofOutputs];
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofOutputs; j++) {
				normOutputs[i][j] = ((2*(outputs[i][j] - xmin)) / (xmax - xmin)) - 1.0;
			}
		}
		
		normalData = new double [numberofPatterns] [numberofInputs+numberofOutputs];
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofInputs+numberofOutputs; j++) {
				if (j<numberofInputs)
					normalData [i][j] = normInputs [i][j];
				else 
					normalData [i][j] = normOutputs [i][j-numberofInputs];
			}
		}
		
	}
	
	
	public void print() throws IOException {
		String fileName = "normdata.txt";  // input for MLP is normalised diff between original and inflated coordinates.
		FileWriter fw = new FileWriter( fileName);
		PrintWriter pw = new PrintWriter( fw);
	
		for (int i=0; i<numberofPatterns; i++) {
			for (int j=0; j<numberofInputs+numberofOutputs; j++) {
				pw.print(normalData[i][j]+"  ");
			}
			pw.println();	
		}
		pw.close();
		fw.close();
		
	}
	
		
	public static void main (String [] args) throws IOException  {
		Normalizer n = new Normalizer ("data.txt");
		//n.readData();
		n.normalize();	
		n.print();
		n.dividePatterns("normdata.txt",20);	
	}
}

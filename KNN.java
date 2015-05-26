import java.util.Scanner;
import java.io.*;

public class KNN
{
	public native int KNN_search(int ref_num, int query_num, int dims, int kn);
	
	public static void main(String[] args) 
    {
        System.out.println("In Main");
	//System.loadLibrary("libKNN.so");		
	System.load("/home/karthik/Heterospark/Heterospark/libKNN.so");		
        KNN knn = new KNN();
		int ref_num     = 4096;   // Reference point number, max=65535
		int query_num   = 4096;   // Query point number, max=65535
		int dims        = 32;     // Dimension of points
		int kn          = 20;	 //  Nearest neighbors to consider
        System.out.println("call search");

		knn.KNN_search(ref_num, query_num, dims, kn);			
			
	}
}

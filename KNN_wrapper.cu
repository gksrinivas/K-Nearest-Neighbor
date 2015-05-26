#include <stdio.h>
#include <stdlib.h>
#include "KNN.h"
#include <time.h>
//#include <jni.h>
#include "cuda.h"
#include "knn_cuda_with_indexes.cu"

JNIEXPORT jint JNICALL Java_KNN_KNN_1search
  (JNIEnv *env, jobject obj, jint ref_nb, jint query_nb, jint dim, jint k) {
	// Variables and parameters
    float* ref;                 // Pointer to reference point array
    float* query;               // Pointer to query point array
    float* dist;                // Pointer to distance array
	int*   ind;                 // Pointer to index array
	int    iterations = 100;
	int    i;
	
	// Memory allocation
	ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
	query  = (float *) malloc(query_nb * dim * sizeof(float));
	dist   = (float *) malloc(query_nb * k * sizeof(float));
	ind    = (int *)   malloc(query_nb * k * sizeof(float));
	
	// Init 
	srand(time(NULL));
	for (i=0 ; i<ref_nb   * dim ; i++) ref[i]    = (float)rand() / (float)RAND_MAX;
	for (i=0 ; i<query_nb * dim ; i++) query[i]  = (float)rand() / (float)RAND_MAX;
	
	// Variables for duration evaluation
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	
	// Display informations
	printf("Number of reference points      : %6d\n", ref_nb  );
	printf("Number of query points          : %6d\n", query_nb);
	printf("Dimension of points             : %4d\n", dim     );
	printf("Number of neighbors to consider : %4d\n", k       );
	printf("Processing kNN search           :"                );
	
	// Call kNN search CUDA
	cudaEventRecord(start, 0);
	for (i=0; i<iterations; i++)
		knn(ref, ref_nb, query, query_nb, dim, k, dist, ind);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time/1000, iterations, elapsed_time/(iterations*1000));
	
	// Destroy cuda event object and free memory
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(ind);
	free(dist);
	free(query);
	free(ref);

return 0;
  }
  

int main() {
return 0;
}

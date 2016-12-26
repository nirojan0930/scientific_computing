#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include <errno.h>

#define GET_TIME(x);  if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }


#define VECTOR_SIZE 10000000  
// #define VECTOR_SIZE 50000000
// #define VECTOR_SIZE 100000000  

// for the calculations of cuda 
#define THREADS_PER_BLOCK 256
#define CALCS_PER_THREAD 50

// for PThreads
#define MAX_PTHREADS 8


#ifdef DP                    	// for the Double precision 
typedef double Real;
#else				// for the single precision
typedef float Real;
#endif


 float elapsed_time_msec(struct timespec *begin, struct timespec *end, long *sec, long *nsec)
 {
  if (end->tv_nsec < begin->tv_nsec) {
    *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
    *sec = end->tv_sec - begin->tv_sec -1;
  }
  else {
    *nsec = end->tv_nsec - begin->tv_nsec;
    *sec = end->tv_sec - begin->tv_sec;
  }
  return (float) (*sec) * 1000 + ((float) (*nsec)) / 1000000;
}


 void init_vector(Real vector[]){
 	for(long i=0;i<VECTOR_SIZE;i++){
    vector[i]=(rand() / (float) RAND_MAX)+1;   // Initializes a given vector with values between 1 and 2
 		
  }
}


 Real serial_calculation(Real vector1[], Real vector2[]){

 	Real result;

 	for(long i=0;i<VECTOR_SIZE;i++){
 		result += vector1[i] * vector2[i];	//serial calculation of the vector dot product
 	}

 	return result;
 }


//defien the the parameters with struct type for the pthread callings
 struct pthread_arg {
 	int thread_id;
 	int no_of_threads;
 	Real *vector1;
 	Real *vector2;
 };


 void *pthread_calculation(void *arguments){

 	struct pthread_arg *args = (struct pthread_arg *)arguments;
 	int no_of_threads = args -> no_of_threads;
	int thread_id = args -> thread_id;       //obtain the value of thread id
	Real *vector1=args -> vector1;
	Real *vector2=args -> vector2;

	Real *result = (Real *)malloc(sizeof(Real));
	*result=0;

	// calculate the range to be multiplied
	int chunk_size = VECTOR_SIZE/no_of_threads;
	int lowerbound=chunk_size*thread_id;			// lowest index to be calculated
	int upperbound=lowerbound+chunk_size-1;	// highest index to be calculated

	for(int i=lowerbound;i<=upperbound;i++){
		*result+=vector1[i]*vector2[i];
	}

	// allocate the leftover vector elements to master
	if(0==thread_id && (0!=VECTOR_SIZE%no_of_threads)){
		for(int i=chunk_size*no_of_threads;i<=VECTOR_SIZE;i++){
			*result+=vector1[i]*vector2[i];
		}

	}

   	pthread_exit((void *)result);     //return the in count

   }

//Vector dot product code for a single CUDA thread
 __global__ void cuda_calculation(Real *vector1, Real *vector2, Real *result) {


 	unsigned long start_point = threadIdx.x + blockDim.x * blockIdx.x;

	// calculate the range to be multiplied
  long lowerbound=start_point*CALCS_PER_THREAD;

  long upperbound=lowerbound+CALCS_PER_THREAD-1;

  // Don't try to calculate beyond vector size
  if(upperbound>=VECTOR_SIZE){
    upperbound=VECTOR_SIZE-1;
  }

  __shared__ Real cache[THREADS_PER_BLOCK] ;


  // initialize the cache
  if(threadIdx.x==0){
    for(int count=0;count<THREADS_PER_BLOCK;count++){
      cache[count]=0;
    }

  }


  Real sum=0.0f;
  
  for(long index=lowerbound;index<=upperbound;index++){
    sum += vector1[index]*vector2[index];
  }

  // Wait till master has finished clearing the cache
  __syncthreads();

  // store the sum
  cache[threadIdx.x] = sum;

  sum=0.0f;

  // should wait till everyone has finished computing
  __syncthreads();

   // take the sum of the elements
  if(threadIdx.x==0){
    
    for(int count=0;count<THREADS_PER_BLOCK;count++){
      sum += cache[count];
    }
    result[blockIdx.x]=sum;
  }


}

int main(int argc, char const *argv[])
{

  struct timespec t1, t2;
  long sec, nsec;
  float comp_time;  // in milli seconds

  // Initialize the random seed
  srand(time(NULL));

	// check the inputs and set the mode
  if(argc<2){
   printf("Incorrect usage of the code!\n");
 }
	// initialize the vectors
 static Real vector1[VECTOR_SIZE];
 static Real vector2[VECTOR_SIZE];
 init_vector(vector1);
 init_vector(vector2);

	// if a serial execution is needed
 if(0==strcmp(argv[1],"-s")){

   GET_TIME(t1);

   Real result= serial_calculation(vector1,vector2);

   GET_TIME(t2);

   comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
   printf("[SERIAL] N=%d: CPU Time(ms)=%.2f \n", VECTOR_SIZE, comp_time);
	#ifdef DP	
	printf("CPU SERIAL result(DP) = %20.18f\n", result); 
	#else
	printf("CPU SERIAL result = %f\n", result);
	#endif
 }
	// if a parallel execution is needed
 else if(0==strcmp(argv[1],"-p")){

   int num_of_threads;
		// check whether the given # of threads is valid
   if(argc <3){
    printf("Incorrect usage of the code!\n");
    return -1;
  }
  num_of_threads=atoi(argv[2]);
  if(num_of_threads>MAX_PTHREADS){
    printf("[ERROR-PTHREADS] - Only up to 8 threads can be created\n");
    return -1;
  }

  pthread_t threads[num_of_threads];
  int rc;
  long t;
  void *status;
  Real result=0;

  GET_TIME(t1);

  //initialize the threads
  for(t=0;t<num_of_threads;t++){
    struct pthread_arg* args=(\
     struct pthread_arg*)malloc(sizeof *args);

    args->no_of_threads=num_of_threads;
    args->thread_id=t;
    args-> vector1=vector1;
    args-> vector2=vector2;

    rc = pthread_create(&threads[t], NULL, pthread_calculation,(void *)args);
    if (rc){
     printf("ERROR; return code from pthread_create() is %d\n", rc);
     exit(-1);
   }
 }

   		//join the threads
 for(t=0;t<num_of_threads;t++){
  pthread_join(threads[t], &status);
            result+=*(Real*)status;            //keep track of the total in count

          }

          GET_TIME(t2);

          comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
          printf("[PTHREAD] N=%d: CPU Time(ms)=%.2f \n", VECTOR_SIZE, comp_time);
      
	  #ifdef DP	
	  printf(" PTHREAD output result(DP) = %20.18f\n", result); 
	  #else
	  printf(" PTHREAD output result = %f\n", result);
	  #endif


// if verification needed
          if(argc == 4 &&0==strcmp(argv[3],"-v")){
            GET_TIME(t1);

            Real result= serial_calculation(vector1,vector2);

            GET_TIME(t2);

            comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
            printf("[SERIAL] N=%d: CPU Time(ms)=%.2f \n", VECTOR_SIZE, comp_time);
            #ifdef DP	
	  printf("SERIAL output result(DP) = %20.18f\n", result); 
	  #else
	  printf("SERIAL output result = %f\n", result);
	  #endif
          }

        }

        


	// if CUDA execution is needed
        else if(0==strcmp(argv[1],"-c")){

		//Allocate vectors in device memory
          size_t size = VECTOR_SIZE * sizeof(Real);
          Real* _vector1;

          GET_TIME(t1);

          gpuErrchk(cudaMalloc((void**) &_vector1, size));

          Real* _vector2;
          gpuErrchk(cudaMalloc((void**) &_vector2, size));

		//copy vectors from host memory to device memory
          cudaMemcpy(_vector1, vector1,size,cudaMemcpyHostToDevice);
          cudaMemcpy(_vector2, vector2,size,cudaMemcpyHostToDevice);


          long num_of_grids=(VECTOR_SIZE/(THREADS_PER_BLOCK*CALCS_PER_THREAD))+1;

// Allocate memory for results in the host memory
          Real results[num_of_grids]; 

          Real* _results;
          size_t result_size = sizeof(Real)*num_of_grids;
          gpuErrchk(cudaMalloc((void**) &_results, result_size));

		// carry out the calculations
          cuda_calculation\
          <<<num_of_grids,THREADS_PER_BLOCK>>>(_vector1,_vector2,_results);

		// copy the results back from the device memory to host memory
          cudaMemcpy(results,_results, sizeof(Real)*num_of_grids,cudaMemcpyDeviceToHost);

          GET_TIME(t2);

          comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
          printf("[CUDA] N=%d: CPU Time(ms)=%.2f \n", VECTOR_SIZE, comp_time);
		// free device memory
          cudaFree(_vector1);
          cudaFree(_vector2);
          cudaFree(_results);


		// calculate the final result
          Real result=0;
          for(long i=0;i<num_of_grids;i++){
            result+=results[i];

    		// }
          }

          #ifdef DP	
	  printf("CUDA output result(DP) = %20.18f\n", result); 
	  #else
	  printf("CUDA output result = %f\n", result);
	  #endif

        // if verification needed
          if(argc == 3 &&0==strcmp(argv[2],"-v")){
            GET_TIME(t1);

            Real result= serial_calculation(vector1,vector2);

            GET_TIME(t2);

            comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
            printf("[SERIAL] N=%d: CPU Time(ms)=%.2f \n", VECTOR_SIZE, comp_time);
            #ifdef DP	
	  printf("SERIAL output result(DP) = %20.18f\n", result); 
	  #else
	  printf("SERIAL output result = %f\n", result);
	  #endif
          }


        }
        else{
         printf("Incorrect usage of the code!\n");
       }
       return 0;
     }

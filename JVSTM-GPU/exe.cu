
#include <time.h>
#include "API.cuh"
#include "util.cuh"
#include <unistd.h>

#define KERNEL_DURATION 5
#define DISJOINT 0

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})


__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ int waitMem;

__global__ void bank_kernel(
	int *flag,
	unsigned int seed,
	int prRead,
	unsigned int roSize,
	unsigned int txSize,
	unsigned int dataSize,
	unsigned int threadNum,
	VertionedDataItem* data,
	TXRecord* record,
	TMmetadata* metadata,
	stm_stats_t* stats)
{
	local_metadata txData;
	bool result;

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	long mod = 0xFFFF;
	int rnd;
	int probRead; // = prRead;
	probRead = (prRead / 10.0) * 0xFFFF;

	uint64_t state = seed + id;
	
	int value=0;
	int addr;

	unsigned long int updates = 0, reads = 0;
	unsigned long int commits = 0, aborts = 0;

#if DISJOINT
	int min, max;
	min = dataSize/threadNum*id;
	max = dataSize/threadNum*(id+1)-1;
#endif

	while((*flag & 1)==0)
	{
		waitMem = *flag;
		///////
		//decide whether the thread will do update or read-only tx
		//if(get_lane_id()==0) {
		rnd = RAND_R_FNC(state) & mod;
		//}
		//rnd = __shfl_sync(0xffffffff, rnd, 0);
		///////
		//rnd = (RAND_R_FNC(seed) % 10) +1; 

		do
		{	
			
			TXBegin(*metadata, &txData);
			
			//Read-Only TX
			if(rnd <= probRead)
			{
				value=0;
				for(int i=0; i<roSize && txData.isAborted==false; i++)//for(int i=0; i<roSize && txData.isAborted==false; i++)//
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state) % dataSize;
			#endif
					value+=TXReadOnly(data, i, &txData);
				}
				if(txData.isAborted==true)
				{
					////atomicAdd(&(stats->nbAbortsDataAge), 1);
					aborts++;
					continue;
				}
				//if(value != 10*dataSize)
				//	printf("T%d found an invariance fail: %d\n", id, value);
			}
			//Update TX
			else
			{
/*				for(int i=0; i<max(txSize,roSize) && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					if(i<roSize)
						value = TXRead(data, addr, &txData);
					if(i<txSize)
						TXWrite(data, value+(1), addr, &txData);
*/
				for(int i=0; i<txSize && txData.isAborted==false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state) % dataSize;
			#endif
					value = TXRead(data, addr, &txData); 
					TXWrite(data, value-(1), addr, &txData);	

			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state) % dataSize;
			#endif
					value = TXRead(data, addr, &txData); 
					TXWrite(data, value+(1), addr, &txData);
				}
				if(txData.isAborted==true)
				{
					////atomicAdd(&(stats->nbAbortsDataAge), 1);
					aborts++;
					continue;
				}
			}
			result = TXCommit(id, record, data, metadata, txData, stats);
  			if(!result) {
				aborts++;
			}
		} while(!result);
		////atomicAdd(&(stats->nbCommits), 1);
		commits++;
		if(txData.ws.size==0)
			reads++;
		else
			updates++;		
	}
	stats[id].commits += commits;
	stats[id].aborts += aborts;
	stats[id].reads += reads;
	stats[id].updates += updates;
}


void getKernelOutput(stm_stats_t *stats, unsigned int nb_threads, float totT_ms, unsigned int verbose){
	unsigned long int nb_commits = 0;
	unsigned long int nb_aborts = 0;
	unsigned long int nb_reads = 0;
	unsigned long int nb_updates = 0;
	for(int i=0; i<nb_threads; i++){
		nb_commits += stats[i].commits;
		nb_aborts += stats[i].aborts;
		nb_reads += stats[i].reads;
		nb_updates += stats[i].updates;
	}
	float throughtput = ((float)nb_commits) / totT_ms * 1000.0;
	float abort_rate = ((float)nb_aborts) / totT_ms * 1000.0;
	float read_update_ratio = ((float)nb_reads) / ((float)nb_reads + (float)nb_updates) * 100.0;
	float abort_ratio = ((float)nb_aborts) / ((float)nb_commits + (float)nb_aborts) * 100.0;
	if(verbose != 0){
		printf("#commits: %lu\n", nb_commits);
		printf("#aborts: %lu\n", nb_aborts);
		printf("time: %f\n", totT_ms);
		printf("#throughtput: %f\n", throughtput);
		printf("#abort_rate: %f\n", abort_rate);
		printf("#reads: %lu\n", nb_reads);
		printf("#updates: %lu\n", nb_updates);
		printf("#read_update_ratio: %f\n", read_update_ratio);
		printf("#abort_ratio: %f\n", abort_ratio);
	}
}


int main(int argc, char *argv[])
{
	unsigned int blockNum, threads_per_block, roSize, threadSize, dataSize, seed, verbose;
	int prRead;

	VertionedDataItem *h_data, *d_data;
	TXRecord *records;
	TMmetadata *metadata;

	stm_stats_t *h_stats, *d_stats;

  	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) nb bank accounts               \n"
	  "  2) client config - nb threads     \n"
	  "  3) client config - nb blocks      \n"
	  "  4) prob read TX                   \n"
	  "  5) read TX Size                   \n"
	  "  6) update TX Size                 \n"
	  "  7) verbose		                   \n"
	"";
	const int NB_ARGS = 8;
	int argCnt = 1;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

	seed 				= 1;
	dataSize			= atoi(argv[argCnt++]);
	threads_per_block	= atoi(argv[argCnt++]);
	blockNum		 	= atoi(argv[argCnt++]);
	prRead 				= atoi(argv[argCnt++]);
	roSize 				= atoi(argv[argCnt++]);
	threadSize			= atoi(argv[argCnt++]);
	verbose				= atoi(argv[argCnt++]);

	printf("#accounts: %u\n", dataSize);
	printf("#threads: %u\n", threads_per_block);
	printf("#blocks: %u\n", blockNum);
	printf("#prob_read: %d\n", prRead);
	printf("#ro_size: %u\n", roSize);
	printf("#rw_size: %u\n", threadSize);

#if DISJOINT
	dataSize=100*blockNum*threads_per_block;
#endif
	
	h_stats = (stm_stats_t*)calloc(blockNum*threads_per_block, sizeof(stm_stats_t));
	
	h_data = (VertionedDataItem*)calloc(dataSize,sizeof(VertionedDataItem));

	//Select the GPU Device
	cudaError_t result;
	result = cudaSetDevice(0);
	if(result != cudaSuccess) fprintf(stderr, "Failed to set Device: %s\n", cudaGetErrorString(result));

	//int peak_clk=1;
	//cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
  	//if (err != cudaSuccess) {printf("cuda err: %d at line %d\n", (int)err, __LINE__); return 1;}
	
	result = TXInit(&records, &metadata);
	if(result != cudaSuccess) fprintf(stderr, "Failed TM Initialization: %s\n", cudaGetErrorString(result));

	cudaMalloc((void **)&d_stats, blockNum*threads_per_block*sizeof(stm_stats_t));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));

	result = cudaMalloc((void **)&d_data, dataSize*sizeof(VertionedDataItem));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));

	for(int i=0; i<dataSize; i++)
	{
		h_data[i].head_ptr = 1;
		h_data[i].value[h_data[i].head_ptr] = 10;
	}

	dim3 blockDist(threads_per_block, 1, 1);
	dim3 gridDist(blockNum, 1, 1);

	cudaMemcpy(d_data, h_data, dataSize*sizeof(VertionedDataItem), cudaMemcpyHostToDevice);

	cudaMemcpy(d_stats, h_stats, blockNum*threads_per_block*sizeof(stm_stats_t), cudaMemcpyHostToDevice);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *flag;
  	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
  	*flag = 0;

	cudaEventRecord(start); 
	bank_kernel<<<gridDist, blockDist>>>(flag, seed, prRead, roSize, threadSize, dataSize, blockNum*threads_per_block, d_data, records, metadata, d_stats);
  	cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");
  	
  	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;

	cudaMemcpy(h_stats, d_stats, blockNum*threads_per_block*sizeof(stm_stats_t), cudaMemcpyDeviceToHost);
  	  	
  	getKernelOutput(h_stats, blockNum*threads_per_block, totT_ms, verbose);

	TXEnd(dataSize, h_data, &d_data, &records, &metadata);

	free(h_stats);
	cudaFree(d_stats);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	CUDA_CHECK_ERROR(cudaDeviceReset(), "cudaDeviceReset");
	
	return 0;
}

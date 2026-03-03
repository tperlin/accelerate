
//#include "STM.cuh"
#include "STM.cu"
#include "util.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <time.h>
#include <unistd.h>

#define KERNEL_DURATION 5
#define DISJOINT 0
#define INITIAL_BALANCE 100


__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ int waitMem;


template <typename T> 
__global__ void bank_kernel(
	int *flag,
	unsigned int seed,
	int prRead,
	unsigned int roSize,
	unsigned int txSize,
	unsigned int dataSize,
	unsigned int threadNum,
	struct STMData<T> *stm_data,
	stm_stats_t* stats)
{
	cg::grid_group grid = cg::this_grid();
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int rank = grid.thread_rank();
//int lane = id & 0x1F; // ID da thread no warp
	long mod = 0xFFFF;
	int rnd;
	int probRead; // = prRead;
	probRead = (prRead / 10.0) * 0xFFFF;

	uint64_t state = seed+id;
	
	TX_Data *tx_data = TX_Init(stm_data, id);

	T value = 0;
	int addr, addr1, addr2;

	int need_gc_me; // Indica a necessidade de coleta de lixo para esta thread
	//int need_gc_warp; // Indica a necessidade de coleta de lixo para o warp
	
	long int updates = 0, reads = 0;
	long int commits = 0, aborts = 0;

	//dijoint accesses variables
	int read;
#if DISJOINT
	int min, max;
	int read;
	min = dataSize/threadNum*id;
	max = dataSize/threadNum*(id+1)-1;
#endif

	//while((*flag & 1)==0)
	while(*flag != 2)
	{ 
		waitMem = *flag;
		///////
		//decide whether the thread will do update or read-only tx
		//if(get_lane_id()==0)
		//{
		rnd = RAND_R_FNC(state) & mod;
		//}
		//rnd = __shfl_sync(0xffffffff, rnd, 0);
		///////
		//decide whether the thread will do update or read-only tx
		//rnd = (RAND_R_FNC(seed) % 10) +1;

		do
		{	
			TX_Start(stm_data, tx_data);
			if(rnd <= probRead)
			{
				value = 0;
				for(int i=0; i < roSize && stm_data->tr_state[tx_data->tr_id] != ABORTED; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state) % dataSize;
					//assert(addr>=0 && addr<dataSize);
			#endif
					read = TX_Open_Read(stm_data, tx_data, addr);
					if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
					{
						value += read;
					}
				}
				if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
				{
					TX_abort_tr(stm_data, tx_data);
					aborts++;
					continue;
				}
			}
			
			//Update TX
			else
			{
/*				for(int i=0; i<max(txSize, roSize) && txData.isAborted == false; i++)
				{
			#if DISJOINT					
					addr = RAND_R_FNC(state)%(max-min+1) + min;
			#else
					addr = RAND_R_FNC(state)%dataSize;
			#endif
					if(i < roSize)
						value = TXRead(data, addr, &txData);
					if(i < txSize)
						TXWrite(data, value+(1), addr, &txData);
*/
//result =0;
				for(int i=0; i < txSize && stm_data->tr_state[tx_data->tr_id] != ABORTED; i++)
				{
					addr1 = RAND_R_FNC(state) % dataSize;
					addr2 = RAND_R_FNC(state) % dataSize;
				
					T* ptr1 = TX_Open_Write(stm_data, tx_data, addr1);

					//if(stm_data->tr_state[tx_data->tr_id] != ABORTED)
					if(ptr1 !=0 )
					{
						T* ptr2 = TX_Open_Write(stm_data, tx_data, addr2);
						if(ptr2 !=0 )
						{
							*ptr1 -= 1;
							*ptr2 += 1;
						}
					}
				
				}
				if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
				{
					TX_abort_tr(stm_data, tx_data);
					aborts++;
					continue;
				}
			}
			TX_commit(stm_data, tx_data);
			
			if(stm_data->tr_state[tx_data->tr_id] == ABORTED)
			{
				TX_abort_tr(stm_data, tx_data);
				aborts++;
			}
		} while(stm_data->tr_state[tx_data->tr_id] != COMMITTED);
		
		commits++;
		if(tx_data->write_set.size == 0)
			reads++;
		else
			updates++;

		if (tx_data->next_locator > ((MAX_LOCATORS) / 2)){ // Verifica a necessidade de coleta de lixo nesta thread
			need_gc_me = 1;
		} else{
			need_gc_me = 0;
		}
				
		//need_garbage_collect_warp = __reduce_or_sync(FULL_MASK, need_garbage_collect); 
		//need_gc_warp = __ballot_sync(FULL_MASK, need_gc_me); // Atualiza a necessidade de coleta de lixo para o warp
			
		if( need_gc_me !=0 ){ 
			if( stm_data->need_gc == 0) {
				atomicCAS((int *)&stm_data->need_gc, 0, 1); // Indica a necessidade de coleta de lixo para todo o sistema
			}
		}
			
		//__syncthreads();
		grid.sync();
		if( need_gc_me == 1 || stm_data->need_gc == 1){
			TX_garbage_collect(stm_data, tx_data);
		}
		if(rank == 0) {
			if(*flag != 0) {
				*flag = 2;
			}
		}
		grid.sync();
		if(rank == 0) {
			atomicCAS((int *)&stm_data->need_gc, 1, 0); // Remove a necessidade de coleta de lixo para todo o sistema
		}
		//__syncthreads();
		grid.sync();
	}
	stats[id].commits += commits;
	stats[id].aborts += aborts;
	stats[id].reads += reads;
	stats[id].updates += updates;
}


void getKernelOutput(stm_stats_t *stats, unsigned int nb_threads, float totT_ms, unsigned int verbose)
{
	if(verbose != 0){
		long int nb_commits = 0;
		long int nb_aborts = 0;
		long int nb_reads = 0;
		long int nb_updates = 0;

		for(int i=0; i<nb_threads; i++){
			nb_commits += stats[i].commits;
			nb_aborts += stats[i].aborts;
			nb_reads += stats[i].reads;
			nb_updates += stats[i].updates;
		}

		float throughtput = ((float)nb_commits) / totT_ms * 1000.0;
		float read_update_ratio = ((float)nb_reads) / ((float)nb_reads + (float)nb_updates) * 100.0;
		float abort_ratio = ((float)nb_aborts) / ((float)nb_commits + (float)nb_aborts) * 100.0;
		
		printf("#commits: %ld\n", nb_commits);
		printf("#aborts: %ld\n", nb_aborts);
		printf("time: %f\n", totT_ms);
		printf("#throughtput: %f\n", throughtput);
		printf("#reads: %ld\n", nb_reads);
		printf("#updates: %ld\n", nb_updates);
		printf("#read_update_ratio: %f\n", read_update_ratio);
		printf("#abort_ratio: %f\n", abort_ratio);
	}
}


template <typename T> 
__device__ __host__ int consistency_check(struct STMData<T> *stm_data, T value)
{
	T total = 0;
	
	for (int i = 0; i < stm_data->n_objects; i++)
	{
		int addr_locator = stm_data->vboxes[i];
		Locator *loc = &stm_data->locators[addr_locator];
		if (stm_data->tr_state[loc->owner] == COMMITTED)
		{
			total += stm_data->locators_data[addr_locator*2];
		} else
		{
			total += stm_data->locators_data[(addr_locator*2)+1];
		}
		assert(stm_data->tr_state[loc->owner] == COMMITTED || stm_data->tr_state[loc->owner] == ABORTED);
	}
	
	int nb_objects = stm_data->n_objects;
	T expected_total = nb_objects * value;
	
	if (total != expected_total)
	{
		printf("Consistency fail!\n");
		printf("Total: %d\n", total);
		printf("Expected total: %d\n", expected_total);
		return 0;
	} else
	{
		printf("Consistency OK! Total: %d Expected: %d\n", total, expected_total);
	}

	for(int i=0; i<stm_data->num_tr; i++){
		if(stm_data->tr_state[i] != COMMITTED){
			printf("Erro status final\n");
		}
	}
	
	return 1;
}


int main(int argc, char *argv[])
{
	unsigned int blockNum, threads_per_block, roSize, threadSize, dataSize, seed, verbose;
	int prRead;
	 
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
	
	if (argc != NB_ARGS)
	{
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
	printf("#w_size: %u\n", threadSize);

#if DISJOINT
	dataSize=100*blockNum*threads_per_block;
#endif

	int num_objects = dataSize;
	int num_locators = MAX_LOCATORS;
	int num_tx = threads_per_block * blockNum;

	h_stats = (stm_stats_t*) calloc(num_tx, sizeof(stm_stats_t));
	
	CUDA_CHECK_ERROR(cudaSetDevice(0), "Failed to set Device");

	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_stats, num_tx * sizeof(stm_stats_t)), "Could not alloc stats");
	
	dim3 blockDist(threads_per_block, 1, 1);
	dim3 gridDist(blockNum, 1, 1);
	
	cudaMemcpy(d_stats, h_stats, num_tx*sizeof(stm_stats_t), cudaMemcpyHostToDevice);

	struct OFGSTM<int> stm;

	struct STMData<int> *stm_data = STM_start(&stm, num_objects, num_tx, num_locators); 
	init_objects(stm_data, num_objects, INITIAL_BALANCE);
	init_locators(stm_data, num_tx, num_locators);

	struct STMData<int> *d_stm_data = STM_copy_to_device(&stm);

	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *flag;
	CUDA_CHECK_ERROR(cudaMallocManaged(&flag, sizeof(int)), "Could not alloc");
	*flag = 0;

	void* bank_kernel_args[] = {
		(void*) &flag,
		(void*) &seed,
		(void*) &prRead,
		(void*) &roSize,
		(void*) &threadSize,
		(void*) &dataSize,
		(void*) &num_tx,
		(void*) &d_stm_data,
		(void*) &d_stats
	};

	//int sharedMemorySize = num_tx * sizeof(unsigned int);

	cudaEventRecord(start); 
	cudaLaunchCooperativeKernel((void*)bank_kernel<int>, gridDist, blockDist, bank_kernel_args);
 	cudaEventRecord(stop);
		
	//sleep for a set time to let the kernel run
	sleep(KERNEL_DURATION);
	//send termination message to the kernel and wait for a return
	__atomic_fetch_add(flag, 1, __ATOMIC_ACQ_REL);

	CUDA_CHECK_ERROR(cudaEventSynchronize(stop), "in kernel");

	cudaEventElapsedTime(&tKernel_ms, start, stop);
	totT_ms += tKernel_ms;

	CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	cudaMemcpy(h_stats, d_stats, num_tx * sizeof(stm_stats_t), cudaMemcpyDeviceToHost);
		
	getKernelOutput(h_stats, num_tx, totT_ms, verbose);
	
	STM_copy_from_device(&stm);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	consistency_check(stm_data, INITIAL_BALANCE);

	STM_stop(&stm);

	free(h_stats);
	cudaFree(d_stats);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	CUDA_CHECK_ERROR(cudaDeviceReset(), "cudaDeviceReset");
	
	return 0;
}

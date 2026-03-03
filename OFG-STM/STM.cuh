#ifndef STM_API_H
#define STM_API_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <stdint.h>

#define BACKOFF         10
#define WRITE_SET_SIZE	200
#define READ_SET_SIZE	200
#define MAX_LOCATORS    10000

#define ACTIVE      1
#define COMMITTED   2
#define ABORTED     3

#define CUDA_MAX_UNSIGNED_INT ~0	// ...1111 Valor maximo que cabe em um inteiro sem sinal
#define TX_STATUS_ID_INC_VAL 4	// ...0000100 Valor de cada incremento da variavel id do status da tx, os ultimos 2 bits serao usados para o status
#define TX_STATUS_MASK 3		// ...000011 Sao usados os 2 ultimos bits para o status

#define INC_TX_STATUS_ID(id) (id<CUDA_MAX_UNSIGNED_INT? id+TX_STATUS_ID_INC_VAL: 0) // Incremento da variavel id do status da tx
#define EXTRACT_TX_STATUS(tx_status) (tx_status & TX_STATUS_MASK) // Extrai apenas o status da tx
#define MAKE_TX_STATUS(id, status) (id | status)	// Junta o id e status da tx


typedef struct Locator_{
    volatile int owner;
    int object;
} Locator;

typedef struct ReadSet_{
    int size;
    int locators[READ_SET_SIZE];
    int objects[READ_SET_SIZE];
    int values[READ_SET_SIZE];
} ReadSet;

typedef struct WriteSet_{
    int size;
    int locators[WRITE_SET_SIZE];
    int objects[WRITE_SET_SIZE];
} WriteSet;

typedef struct TX_Data_{
    int tr_id;
    unsigned int status_id;
    ReadSet read_set;
    WriteSet write_set;
    int locator_queue[MAX_LOCATORS];
    int used_locators[MAX_LOCATORS];
    int next_locator;
    int n_aborted;
    int n_committed; // maximum 1
    int cm_enemy;
    int cm_aborts;
    int cm_enemies[WRITE_SET_SIZE];
    int enemies_size;
} TX_Data;

template <typename T> struct STMData{
    int n_objects;
    volatile int *vboxes;
    volatile unsigned int *tr_state;
    Locator *locators;
    T *locators_data;
    int num_locators;
    int num_tr;
    TX_Data *tx_data;
};


template <typename T> struct OFGSTM{
    struct STMData<T> *h_stm_data; 
    struct STMData<T> *d_stm_data;
    int n_objects;
	int num_tr;
	int num_locators;
    TX_Data *h_tx_data;
    TX_Data *d_tx_data;
    int *h_vboxes;
    int *d_vboxes;
    unsigned int *h_tr_state;
    unsigned int *d_tr_state;
    Locator *h_locators;
    Locator *d_locators;
    T *h_locators_data;
    T *d_locators_data;
};

typedef struct stm_stats_struct {
	long int commits;
	long int aborts;
	long int updates;
	long int reads;
} stm_stats_t;


template <typename T> struct STMData<T>* STM_start(struct OFGSTM<T> *stm, int numObjects, int numTransactions, int numLocators);
template <typename T> void STM_stop(struct OFGSTM<T> *stm);
template <typename T> struct STMData<T>* STM_copy_to_device(struct OFGSTM<T> *stm);
template <typename T> struct STMData<T>* STM_copy_from_device(struct OFGSTM<T> *stm);
template <typename T> void init_locators(struct STMData<T> *stm_data, int num_tx, int num_locators);
template <typename T> void init_objects(struct STMData<T> *stm_data, int num_objects, T value);

template <typename T> __device__ TX_Data* TX_Init(struct STMData<T> *stm_data, int tx_id);
template <typename T> __device__ void TX_Start(struct STMData<T> *stm_data, TX_Data *d);
template <typename T> __device__ int TX_new_locator(struct STMData<T> *stm_data, TX_Data *tx_data);
template <typename T> __device__ int TX_validate_readset(struct STMData<T> *stm_data, TX_Data *tx_data);
template <typename T> __device__ int TX_commit(struct STMData<T> *stm_data, TX_Data *tx_data);
template <typename T> __device__ T* TX_Open_Write(struct STMData<T> *stm_data, TX_Data *tx_data, int object);
template <typename T> __device__ T TX_Open_Read(struct STMData<T> *stm_data, TX_Data *tx_data, int object);
template <typename T> __device__ void TX_abort_tr(struct STMData<T> *stm_data, TX_Data *tx_data);
template <typename T> __device__ int TX_contention_manager(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy);
template <typename T> __device__ void TX_garbage_collect(struct STMData<T> *stm_data, TX_Data *tx_data);


#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

#endif


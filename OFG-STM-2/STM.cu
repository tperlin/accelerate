#include "STM.cuh"

#define FULL_MASK 0xffffffff

/*Contention Manager*/
#define CM_1 1
#define CM_2 2
#define CM_3 3
#define CM_4 4
#define CM_5 5
#define CM_6 6
#define CM_7 7
#define CM_8 8
#define CM_9 9

#define CM CM_4

int tr_id_gen = 0;


template <typename T>
struct STMData<T> *STM_start(struct OFGSTM<T> *stm, int numObjects, int numTransactions, int numLocators)
{
	printf("Contention Manager: %d\n", CM);

	stm->h_stm_data = (struct STMData<T> *)malloc(sizeof(struct STMData<T>));
	if (stm->h_stm_data == NULL)
	{
		printf("Erro na alocacao de memoria\n");
		return NULL;
	}

	stm->n_objects = numObjects;
	stm->h_stm_data->n_objects = stm->n_objects;

	stm->h_vboxes = (int *)malloc(numObjects * sizeof(int));
	if (stm->h_vboxes == NULL)
	{
		printf("Erro na alocacao de memoria\n");
		return NULL;
	}
	stm->h_stm_data->vboxes = stm->h_vboxes;
	printf("Number of objects: %d\n", numObjects);

	stm->num_tr = numTransactions;
	stm->h_stm_data->num_tr = stm->num_tr;

	stm->h_tx_data = (TX_Data *)malloc(numTransactions * sizeof(TX_Data));
	if (stm->h_tx_data == NULL)
	{
		printf("Erro na alocacao de memoria\n");
		return NULL;
	}
	for (int i = 0; i < numTransactions; i++)
	{
		stm->h_tx_data[i].tr_id = i;
	}
	stm->h_stm_data->tx_data = stm->h_tx_data;
	printf("Number of transactions: %d\n", numTransactions);
	printf("Total transactions size: %lu MB\n", (numTransactions * sizeof(TX_Data)) / 1048576);

	stm->h_tr_state = (unsigned int *)malloc((numTransactions + 2) * sizeof(unsigned int));
	if (stm->h_tr_state == NULL)
	{
		printf("Erro na alocacao de memoria\n");
		return NULL;
	}
	for (int i = 0; i < numTransactions; i++)
	{
		stm->h_tr_state[i] = ABORTED;
	}
	stm->h_stm_data->tr_state = stm->h_tr_state;
	stm->h_stm_data->tr_state[numTransactions] = COMMITTED;
	stm->h_stm_data->tr_state[numTransactions + 1] = ABORTED;

	stm->num_locators = numLocators;
	stm->h_stm_data->num_locators = stm->num_locators;
	printf("Number of locators: %d\n", numLocators);

	stm->h_locators = (Locator *)malloc((numObjects + (numLocators * numTransactions)) * sizeof(Locator));
	if (stm->h_locators == NULL)
	{
		printf("Erro na alocacao de memoria\n");
		return NULL;
	}
	stm->h_stm_data->locators = stm->h_locators;
	printf("Total locators: %d\n", (numObjects + (numLocators * numTransactions)));
	printf("Total locators size: %lu MB\n", (numObjects + (numLocators * numTransactions)) * sizeof(Locator) / 1048576);

	stm->h_locators_data = (T *)malloc(((2 * numObjects) + (2 * numLocators * numTransactions)) * sizeof(T));
	if (stm->h_locators_data == NULL)
	{
		printf("Erro na alocacao de memoria\n");
		return NULL;
	}
	stm->h_stm_data->locators_data = stm->h_locators_data;
	printf("Total locators data: %d\n", (2 * numObjects + (2 * numLocators * numTransactions)));
	printf("Total locators data size: %lu MB\n", (2 * numObjects + (2 * numLocators * numTransactions)) * sizeof(T) / 1048576);
	printf("Total locators global size: %lu MB\n",
		   (((numObjects + (numLocators * numTransactions)) * sizeof(Locator)) +
			((2 * numObjects + (2 * numLocators * numTransactions)) * sizeof(T))) /
			   1048576);

	return stm->h_stm_data;
}


template <typename T>
void STM_stop(struct OFGSTM<T> *stm)
{
	free(stm->h_stm_data);
	cudaFree(stm->d_stm_data);
	free(stm->h_vboxes);
	cudaFree(stm->d_vboxes);
	free(stm->h_tx_data);
	cudaFree(stm->d_tx_data);
	free(stm->h_tr_state);
	cudaFree(stm->d_tr_state);
	free(stm->h_locators);
	cudaFree(stm->d_locators);
	free(stm->h_locators_data);
	cudaFree(stm->d_locators_data);
}


template <typename T>
struct STMData<T> *STM_copy_from_device(struct OFGSTM<T> *stm)
{
	int numObjects = stm->n_objects;
	int numTransactions = stm->num_tr;
	int numLocators = stm->num_locators;

	CUDA_CHECK_ERROR(cudaMemcpy(stm->h_stm_data, stm->d_stm_data, sizeof(struct STMData<T>), cudaMemcpyDeviceToHost), "Error copy to device stm data");

	CUDA_CHECK_ERROR(cudaMemcpy(stm->h_vboxes, stm->d_vboxes, numObjects * sizeof(int), cudaMemcpyDeviceToHost), "Error copy to device vboxes");
	stm->h_stm_data->vboxes = stm->h_vboxes;

	CUDA_CHECK_ERROR(cudaMemcpy(stm->h_tr_state, stm->d_tr_state, (numTransactions + 2) * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Error copy to device tr state");
	stm->h_stm_data->tr_state = stm->h_tr_state;

	CUDA_CHECK_ERROR(cudaMemcpy(stm->h_locators, stm->d_locators, (numObjects + (numLocators * numTransactions)) * sizeof(Locator), cudaMemcpyDeviceToHost), "Error copy to device locators");
	stm->h_stm_data->locators = stm->h_locators;

	CUDA_CHECK_ERROR(cudaMemcpy(stm->h_locators_data, stm->d_locators_data, ((2 * numObjects) + (2 * numLocators * numTransactions)) * sizeof(T), cudaMemcpyDeviceToHost), "Error copy to device locators data ");
	stm->h_stm_data->locators_data = stm->h_locators_data;

	CUDA_CHECK_ERROR(cudaMemcpy(stm->h_tx_data, stm->d_tx_data, numTransactions * sizeof(TX_Data), cudaMemcpyDeviceToHost), "Error copy to device tx data ");
	stm->h_stm_data->tx_data = stm->h_tx_data;

	return stm->h_stm_data;
}


template <typename T>
struct STMData<T> *STM_copy_to_device(struct OFGSTM<T> *stm)
{
	int numObjects = stm->n_objects;
	int numTransactions = stm->num_tr;
	int numLocators = stm->num_locators;

	struct STMData<T> *meta_data = (struct STMData<T> *)malloc(sizeof(struct STMData<T>));
	meta_data->n_objects = numObjects;
	int *d_vboxes;
	CUDA_CHECK_ERROR(cudaMalloc((void **)&d_vboxes, numObjects * sizeof(int)), "Error malloc vboxes");
	CUDA_CHECK_ERROR(cudaMemcpy(d_vboxes, stm->h_vboxes, numObjects * sizeof(int), cudaMemcpyHostToDevice), "Error mem copy vboxes");
	meta_data->vboxes = d_vboxes;
	stm->d_vboxes = d_vboxes;

	unsigned int *d_tr_state;
	CUDA_CHECK_ERROR(cudaMalloc((void **)&d_tr_state, (2 + numTransactions) * sizeof(unsigned int)), "Error malloc tr state");
	CUDA_CHECK_ERROR(cudaMemcpy(d_tr_state, stm->h_tr_state, ((2 + numTransactions) * sizeof(unsigned int)), cudaMemcpyHostToDevice), "Error copy tr state");
	meta_data->tr_state = d_tr_state; // 1 for the always committed Tr and 1 for the always aborted
	stm->d_tr_state = d_tr_state;

	T *d_locators_data;
	CUDA_CHECK_ERROR(cudaMalloc((void **)&d_locators_data, ((2 * numObjects) + (2 * numLocators * numTransactions)) * sizeof(T)), "Error malloc locators data");
	CUDA_CHECK_ERROR(cudaMemcpy(d_locators_data, stm->h_locators_data, ((2 * numObjects) + (2 * numLocators * numTransactions)) * sizeof(T), cudaMemcpyHostToDevice), "Error copy locators data");
	meta_data->locators_data = d_locators_data;
	stm->d_locators_data = d_locators_data;

	Locator *d_locators;
	CUDA_CHECK_ERROR(cudaMalloc((void **)&d_locators, (numObjects + (numLocators * numTransactions)) * sizeof(Locator)), "Error malloc locators");
	CUDA_CHECK_ERROR(cudaMemcpy(d_locators, stm->h_locators, (numObjects + (numLocators * numTransactions)) * sizeof(Locator), cudaMemcpyHostToDevice), "Error copy locators");
	meta_data->locators = d_locators;
	stm->d_locators = d_locators;

	TX_Data *d_tx_data;
	CUDA_CHECK_ERROR(cudaMalloc((void **)&d_tx_data, numTransactions * sizeof(TX_Data)), "Error malloc tx  data");
	CUDA_CHECK_ERROR(cudaMemcpy(d_tx_data, stm->h_tx_data, numTransactions * sizeof(TX_Data), cudaMemcpyHostToDevice), "Error copy tx data");
	meta_data->tx_data = d_tx_data;
	stm->d_tx_data = d_tx_data;

	meta_data->num_locators = numLocators;
	meta_data->num_tr = numTransactions;

	struct STMData<T> *d_stm_data;
	CUDA_CHECK_ERROR(cudaMalloc((void **)&d_stm_data, sizeof(struct STMData<T>)), "Error malloc stm data");
	CUDA_CHECK_ERROR(cudaMemcpy(d_stm_data, meta_data, sizeof(struct STMData<T>), cudaMemcpyHostToDevice), "Error copy stm data");
	stm->d_stm_data = d_stm_data;

	free(meta_data);

	return d_stm_data;
}


template <typename T>
__device__ TX_Data *TX_Init(struct STMData<T> *stm_data, int tx_id)
{
	TX_Data *d = &stm_data->tx_data[tx_id];
	d->tr_id = tx_id;
	// assert(stm_data->tx_data[tx_id].tr_id == d->tr_id);

	d->status_id = 0;
	d->next_locator = 0;
	d->read_set.size = 0;
	d->write_set.size = 0;
	d->n_aborted = 0;
	d->n_committed = 0;
	d->cm_enemy = -1;
	d->cm_aborts = 0;
	stm_data->tr_state[d->tr_id] = ABORTED;

	for (int i = 0; i < stm_data->num_locators; i++)
	{
		d->locator_queue[i] = (tx_id * stm_data->num_locators) + i;
	}
	// assert(stm_data->tr_state[d->tr_id] == ABORTED);

	return d;
}


template <typename T>
__device__ void TX_Start(struct STMData<T> *stm_data, TX_Data *d)
{
	d->read_set.size = 0;
	d->write_set.size = 0;
	// d->n_aborted = 0;
	// d->n_committed = 0;

	unsigned int status = EXTRACT_TX_STATUS(stm_data->tr_state[d->tr_id]);
	if (status == COMMITTED)
	{
		d->enemies_size = 0;
		d->cm_enemy = -1;
		d->cm_aborts = 0;
	}
	// assert(stm_data->tx_data[d->tr_id].tr_id == d->tr_id);
	// assert(stm_data->tr_state[d->tr_id] == ABORTED || stm_data->tr_state[d->tr_id] == COMMITTED);

	__threadfence();
	d->status_id = INC_TX_STATUS_ID(d->status_id);
	stm_data->tr_state[d->tr_id] = MAKE_TX_STATUS(d->status_id, ACTIVE);
}


template <typename T>
__device__ void TX_garbage_collect(struct STMData<T> *stm_data, TX_Data *tx_data)
{
	if (tx_data->next_locator > 0)
	{
		int *used_locators = tx_data->used_locators;
		int used_pos = 0;
		tx_data->next_locator--;
		int next = tx_data->next_locator;

		do
		{
			int next_locator = tx_data->locator_queue[next];
			Locator *locator = &stm_data->locators[next_locator];
			// assert(stm_data->locators[next_locator].owner == tx_data->tr_id || stm_data->locators[next_locator].owner == stm_data->num_tr || stm_data->locators[next_locator].owner == stm_data->num_tr+1);

			if (stm_data->vboxes[locator->object] == next_locator)
			{
				used_locators[used_pos] = next_locator;
				used_pos++;
				next--;
			}
			else
			{
				tx_data->locator_queue[tx_data->next_locator] = next_locator;
				tx_data->next_locator--;
				next--;
			}
		} while (next >= 0);
		int pos_queue = tx_data->next_locator;
		tx_data->next_locator++;
		// assert(tx_data -> next_locator == used_pos);
		used_pos--;
		// assert(used_pos == pos_queue);
		while (pos_queue >= 0)
		{
			tx_data->locator_queue[pos_queue] = used_locators[used_pos];
			pos_queue--;
			used_pos--;
		}
	}
}


template <typename T>
__device__ int TX_new_locator(struct STMData<T> *stm_data, TX_Data *tx_data)
{
	int next_locator = tx_data->locator_queue[tx_data->next_locator];

	tx_data->next_locator++;
	if (tx_data->next_locator == MAX_LOCATORS)
	{
		printf("Max locators reached!\n");
		assert(tx_data->next_locator < MAX_LOCATORS);
	}
	// assert(tx_data->next_locator>=0 && tx_data->next_locator<MAX_LOCATORS);
	// assert(next_locator >=0 && next_locator < stm_data->num_locators * (tx_data->tr_id+1));

	return next_locator;
}


template <typename T>
__device__ int TX_validate_readset(struct STMData<T> *stm_data, TX_Data *tx_data)
{
	ReadSet *read_set = &tx_data->read_set;
	int size = tx_data->read_set.size;

	volatile int locator_addr;
	Locator *locator;
	volatile int locator_owner;
	unsigned int locator_tx_status;

	for (int i = 0; i < size; i++)
	{
		locator_addr = stm_data->vboxes[read_set->objects[i]];
		locator = &stm_data->locators[locator_addr];

		if (locator_addr == read_set->locators[i])
		{
			do
			{
				locator_owner = locator->owner;
				locator_tx_status = stm_data->tr_state[locator_owner];
				//__threadfence();
			} while (locator_owner != locator->owner);

			if (locator_tx_status == COMMITTED)
			{
				if (!(locator_addr * 2 == read_set->values[i]))
				{
					return 0;
				}
				continue;
			}
			else
			{
				if (!((locator_addr * 2) + 1 == read_set->values[i]))
				{
					return 0;
				}
			}
		}
		else
		{
			return 0;
		}
	}
	return 1;
}


template <typename T>
__device__ int TX_commit(struct STMData<T> *stm_data, TX_Data *tx_data)
{
	unsigned int tx_status = stm_data->tr_state[tx_data->tr_id];
	if (tx_status == ABORTED)
	{
		return 0;
	}
	if (TX_validate_readset(stm_data, tx_data))
	{
		__threadfence();
		if (atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], tx_status, COMMITTED) == tx_status)
		{
			for (int i = 0; i < tx_data->write_set.size; i++)
			{
				// assert(
				atomicCAS((int *)&stm_data->locators[tx_data->write_set.locators[i]].owner, tx_data->tr_id, stm_data->num_tr);
				// == tx_data->tr_id);
			}
			tx_data->n_committed++;
			return 1;
		}
	}
	atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], tx_status, ABORTED);
	// assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
	return 0;
}


template <typename T>
__device__ T *TX_Open_Write(struct STMData<T> *stm_data, TX_Data *tx_data, int object)
{
	int addr_locator = stm_data->vboxes[object];
	Locator *locator = &stm_data->locators[addr_locator];
	int locator_owner = locator->owner;

	if (locator_owner == tx_data->tr_id)
	{
		return &stm_data->locators_data[addr_locator * 2];
	}

	while (stm_data->tr_state[tx_data->tr_id] != ABORTED)
	{
		addr_locator = stm_data->vboxes[object];
		locator = &stm_data->locators[addr_locator];

		int addr_new_locator = TX_new_locator(stm_data, tx_data);
		Locator *new_locator = &stm_data->locators[addr_new_locator];
		new_locator->owner = tx_data->tr_id;
		new_locator->object = object;

		unsigned int locator_tx_status;

		do
		{
			locator_owner = locator->owner;
			locator_tx_status = stm_data->tr_state[locator_owner];
			//__threadfence();
		} while (locator_owner != locator->owner);

		unsigned int locator_status = EXTRACT_TX_STATUS(locator_tx_status);
		switch (locator_status)
		{
		case COMMITTED:
			stm_data->locators_data[(addr_new_locator * 2) + 1] = stm_data->locators_data[addr_locator * 2];
			stm_data->locators_data[addr_new_locator * 2] = stm_data->locators_data[(addr_new_locator * 2) + 1];
			break;
		case ABORTED:
			stm_data->locators_data[(addr_new_locator * 2) + 1] = stm_data->locators_data[(addr_locator * 2) + 1];
			stm_data->locators_data[addr_new_locator * 2] = stm_data->locators_data[(addr_new_locator * 2) + 1];
			break;
		case ACTIVE:
			if (TX_contention_manager(stm_data, tx_data, tx_data->tr_id, locator_owner))
			{
				if (stm_data->tr_state[tx_data->tr_id] != ABORTED)
				{
					__threadfence();
					if (atomicCAS((unsigned int *)&stm_data->tr_state[locator_owner], locator_tx_status, ABORTED) == locator_tx_status)
					{
						stm_data->locators_data[(addr_new_locator * 2) + 1] = stm_data->locators_data[(addr_locator * 2) + 1];
						stm_data->locators_data[addr_new_locator * 2] = stm_data->locators_data[(addr_new_locator * 2) + 1];
					}
					else
					{
						// atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], stm_data->tr_state[tx_data->tr_id], ABORTED);
						// assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
						tx_data->next_locator--;
						continue;
					}
				}
				else
				{
					tx_data->next_locator--;
					// assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
					continue;
				}
			}
			else
			{
				atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], stm_data->tr_state[tx_data->tr_id], ABORTED);
				// assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
				tx_data->next_locator--;
				continue;
			}

			break;
		default:
			printf("TX_Write: invalid tr state! Locator %d, Owner %d, state %d\n", addr_locator, locator->owner, stm_data->tr_state[locator->owner]);
			assert(0);
		}

		if (stm_data->tr_state[tx_data->tr_id] != ABORTED)
		{
			__threadfence();
			if (atomicCAS((int *)&stm_data->vboxes[object], addr_locator, addr_new_locator) == addr_locator)
			{

				WriteSet *write_set = &tx_data->write_set;
				int size = tx_data->write_set.size;
				write_set->locators[size] = addr_new_locator;
				write_set->objects[size] = object;
				write_set->size++;

				// if (TX_validate_readset(stm_data, tx_data))
				// {
				return &stm_data->locators_data[addr_new_locator * 2];
				// }
				// else
				// {
				// 	atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], stm_data->tr_state[tx_data->tr_id], ABORTED);
				// 	//assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
				// }
			}
			else
			{
				// atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], stm_data->tr_state[tx_data->tr_id], ABORTED);
				tx_data->next_locator--;
				continue;
			}
		}
		// assert(stm_data->tr_state[tx_data->tr_id] != ACTIVE);
		return 0;
	}
	// assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
	return 0;
}


__device__ inline int is_enemy(TX_Data *tx_data, int enemy)
{
	for (int i = 0; i < tx_data->enemies_size; i++)
	{
		if (tx_data->cm_enemies[i] == enemy)
			return 1;
	}
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_1(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (enemy < me)
		return 1;
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_2(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (tx_data->n_aborted > BACKOFF)
	{
		TX_Data *data_enemy = &stm_data->tx_data[enemy];
		if (data_enemy->write_set.size < tx_data->write_set.size)
			return 1;
		else
			return 0;
	}
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_3(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (tx_data->n_aborted > BACKOFF)
	{
		TX_Data *data_enemy = &stm_data->tx_data[enemy];
		if (data_enemy->write_set.size < tx_data->write_set.size)
		{
			if (data_enemy->n_aborted < tx_data->n_aborted)
				return 1;
			else
				return 0;
		}
		if (data_enemy->n_aborted < tx_data->n_aborted)
			return 1;
		else
			return 0;
	}
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_4(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	TX_Data *data_enemy = &stm_data->tx_data[enemy];
	if (data_enemy->write_set.size < tx_data->write_set.size)
	{
		return 1;
	}
	if (data_enemy->write_set.size == tx_data->write_set.size)
	{
		if (data_enemy->n_aborted < tx_data->n_aborted)
			return 1;
		else
			return 0;
	}
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_5(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (tx_data->n_aborted > 100)
	{
		return 1;
	}
	return 1;
}


template <typename T>
__device__ inline int TX_contention_manager_6(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (tx_data->cm_enemy == enemy)
	{
		tx_data->cm_aborts++;
		if (tx_data->cm_aborts >= 10)
			return 1;
	}
	else
	{
		tx_data->cm_enemy = enemy;
		tx_data->cm_aborts = 0;
	}
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_7(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (tx_data->cm_aborts > 10)
	{
		tx_data->cm_aborts = 0;
		return 1;
	}
	else
	{
		tx_data->cm_aborts++;
		return 0;
	}
}


template <typename T>
__device__ inline int TX_contention_manager_8(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (tx_data->cm_enemy == enemy)
	{
		tx_data->cm_aborts++;
		if (tx_data->cm_aborts >= 100)
		{
			TX_Data *data_enemy = &stm_data->tx_data[enemy];
			if (data_enemy->write_set.size < tx_data->write_set.size)
			{
				return 1;
			}
		}
	}
	else
	{
		tx_data->cm_enemy = enemy;
		tx_data->cm_aborts = 0;
	}
	return 0;
}


template <typename T>
__device__ inline int TX_contention_manager_9(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
	if (is_enemy(tx_data, enemy))
		return 1;
	else
	{
		tx_data->cm_enemies[tx_data->enemies_size] = enemy;
		tx_data->enemies_size++;
	}
	return 0;
}


template <typename T>
__device__ int TX_contention_manager(struct STMData<T> *stm_data, TX_Data *tx_data, int me, int enemy)
{
#if CM == CM_1
	return TX_contention_manager_1(stm_data, tx_data, me, enemy);
#elif CM == CM_2
	return TX_contention_manager_2(stm_data, tx_data, me, enemy);
#elif CM == CM_3
	return TX_contention_manager_3(stm_data, tx_data, me, enemy);
#elif CM == CM_4
	return TX_contention_manager_4(stm_data, tx_data, me, enemy);
#elif CM == CM_5
	return TX_contention_manager_5(stm_data, tx_data, me, enemy);
#elif CM == CM_6
	return TX_contention_manager_6(stm_data, tx_data, me, enemy);
#elif CM == CM_7
	return TX_contention_manager_7(stm_data, tx_data, me, enemy);
#elif CM == CM_8
	return TX_contention_manager_8(stm_data, tx_data, me, enemy);
#elif CM == CM_9
	return TX_contention_manager_9(stm_data, tx_data, me, enemy);
#else
	return TX_contention_manager_4(stm_data, tx_data, me, enemy); // best 4
#endif
}


template <typename T>
__device__ T TX_Open_Read(struct STMData<T> *stm_data, TX_Data *tx_data, int object)
{
	int version;
	volatile int addr_locator = stm_data->vboxes[object];
	Locator *locator = &stm_data->locators[addr_locator];
	volatile int locator_owner;
	unsigned int locator_tx_status;

	do
	{
		locator_owner = locator->owner;
		locator_tx_status = stm_data->tr_state[locator_owner];
		//__threadfence();
	} while (locator_owner != locator->owner);

	switch (EXTRACT_TX_STATUS(locator_tx_status))
	{
	case COMMITTED:
		version = addr_locator * 2;
		break;
	case ABORTED:
		version = (addr_locator * 2) + 1;
		break;
	case ACTIVE:
		version = (addr_locator * 2) + 1;
		break;
	default:
		printf("TX_Read: invalid tr state!\n");
	}

	// if (TX_validate_readset(stm_data, tx_data))
	// {
	ReadSet *read_set = &tx_data->read_set;
	int size = tx_data->read_set.size;
	read_set->locators[size] = addr_locator;
	read_set->values[size] = version;
	read_set->objects[size] = object;
	read_set->size++;
	return stm_data->locators_data[version];
	// }
	// atomicCAS((unsigned int *)&stm_data->tr_state[tx_data->tr_id], stm_data->tr_state[tx_data->tr_id], ABORTED);
	// // assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
	// return 0;
}


template <typename T>
__device__ void TX_abort_tr(struct STMData<T> *stm_data, TX_Data *tx_data)
{

	for (int i = 0; i < tx_data->write_set.size; i++)
	{
		// assert(
		atomicCAS((int *)&stm_data->locators[tx_data->write_set.locators[i]].owner, tx_data->tr_id, (stm_data->num_tr + 1));
		// == tx_data->tr_id);
	}

	// assert(stm_data->tr_state[tx_data->tr_id] == ABORTED);
	//  tx_data-> read_set.size = 0;
	//  tx_data -> write_set.size = 0;
	tx_data->n_aborted++;
}


template <typename T>
void init_objects(struct STMData<T> *stm_data, int num_objects, T value)
{
	stm_data->tr_state[stm_data->num_tr] = COMMITTED;
	int initial_locators = stm_data->num_locators * stm_data->num_tr;
	int pos = 0;
	for (int i = initial_locators; i < (initial_locators + num_objects); i++)
	{
		stm_data->locators_data[2 * i] = value;
		stm_data->locators_data[2 * i + 1] = value;
		stm_data->locators[i].owner = stm_data->num_tr;
		stm_data->vboxes[pos] = i;
		pos++;
	}
}

template <typename T>
void init_locators(struct STMData<T> *stm_data, int num_tx, int num_locators)
{
	int total_locators = num_tx * num_locators;
	for (int i = 0; i < total_locators; i++)
	{
		stm_data->locators_data[2 * i] = 0;
		stm_data->locators_data[2 * i + 1] = 0;
	}
}

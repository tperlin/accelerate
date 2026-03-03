#!/bin/bash
source functions.sh

# Numero de repeticoes
NB_REPEAT=30
# Numero de threads por bloco
NB_THREADS=128
# Numero de blocos
NB_BLOCKS=46
# Probabilidade Leitura/Escrita
RO_PROB=8
# Numero de contas
ACCOUNTS=100000
# Numero de contas alta contencao
NB_ACC_LOW=1000000
# Numero de contas baixa contencao
NB_ACC_HIGH=100000
# Tamanho padrao leitura
RO_SIZE=50
# Tamanho padrao escrita
RW_SIZE=25
# Diferentes STM
STM="OFG-STM OFG-STM-2 AccelerateSTM"
# sequencia objetos
SEQ_OBJ="10 20 30 40 50 60"
#SEQ_OBJ="50 55 60 65 70"
# Sequencia numero de threads
SEQ_THREADS="1 16 32 48 56 64"
# Sequencia probabilidade
SEQ_PROB="1 2 3 4 5 6 7 8 9 10"
# Aplicacao
APP="bank"


# compilar
function compile(){
	cd OFG-STM; pwd; make clean; make; cd ..;
	cd OFG-STM-2; pwd; make clean; make; cd ..;
	cd AccelerateSTM; pwd; make clean; make; cd ..;
}


# Experimento variando-se a probabilidade
# $1: nome do experimento
# $2: numero de contas
# $3: numero de threads
# $4: numero de blocos
# $5: probabilidade de somente leitura
# $6: numero de objetos somente leitura
# $7: numero de objetos leitura e escrita
function exp_bank_prob(){
	local name=$1
	local nb_acc=$2
	local nb_thr=$3
	local nb_bloc=$4
 	#local prob=$5
	local nb_ro=$6
  	local nb_rw=$7
	local exp_file="prob.$name.csv"
	echo "file: $exp_file" # debug
	# cabecalho
	echo "stm, prob, throughput, sd" # debug
 	echo "stm, prob, throughput, sd" > ./results/$exp_file
	for prob in $SEQ_PROB; do
		for stm in $STM; do
			local run_file=$name.$stm.$APP.$nb_acc.$nb_thr.$nb_bloc.$prob.$nb_ro.$nb_rw.txt
			echo $run_file # debug
			echo "" > ./results/$run_file
			for rep in $(seq 1 $NB_REPEAT); do
				# executa aplicacao
				printf "$rep." # debug
   				./$stm/bank $nb_acc $nb_thr $nb_bloc $prob $nb_ro $nb_rw 1 >> ./results/$run_file
			done
			echo "" # debug
  			# processamento resultados
			local tmp_file=$run_file.tmp
			cat ./results/$run_file | filter_throughtput > ./results/$tmp_file
			#cat ./results/$run_file
			cat ./results/$tmp_file
			avg=$(Rscript --vanilla mean_file.R ./results/$tmp_file)
  			sd=$(Rscript --vanilla sd_file.R ./results/$tmp_file)
			rm -f ./results/$tmp_file
			# salva resultado
			echo "$stm, $prob, $avg, $sd" # debug
			echo "$stm, $prob, $avg, $sd" >> ./results/$exp_file
		done
 	done
  	# grafico
  	Rscript --vanilla graph_prob.R ./results/$exp_file ./results/$exp_file
}


# Experimento variando-se o numero de objetos
# $1: nome do experimento
# $2: numero de contas
# $3: numero de threads
# $4: numero de blocos
# $5: probabilidade de somente leitura
# $6: numero de objetos somente leitura
# $7: numero de objetos leitura e escrita
function exp_bank_obj(){
	local name=$1
	local nb_acc=$2
	local nb_thr=$3
	local nb_bloc=$4
 	local prob=$5
	#local nb_ro=$6
  	#local nb_rw=$7
	local exp_file="obj.$name.$prob.csv"
	echo "file: $exp_file" # debug
	# cabecalho
	echo "stm, objects, throughput, sd" # debug
 	echo "stm, objects, throughput, sd" > ./results/$exp_file
	for nb_obj in $SEQ_OBJ; do
		for stm in $STM; do
  			local nb_ro_obj=$(expr $nb_obj \* 2)
			local run_file=$name.$stm.$APP.$nb_acc.$nb_thr.$nb_bloc.$prob.$nb_ro_obj.$nb_obj.txt
			echo $run_file # debug
			echo "" > ./results/$run_file
			for rep in $(seq 1 $NB_REPEAT); do
				# executa aplicacao
				printf "$rep." # debug
   				./$stm/bank $nb_acc $nb_thr $nb_bloc $prob $nb_ro_obj $nb_obj 1 >> ./results/$run_file
			done
			echo "" # debug
  			# processamento resultados
			local tmp_file=$run_file.tmp
			cat ./results/$run_file | filter_throughtput > ./results/$tmp_file
			#cat ./results/$run_file
			cat ./results/$tmp_file
			avg=$(Rscript --vanilla mean_file.R ./results/$tmp_file)
  			sd=$(Rscript --vanilla sd_file.R ./results/$tmp_file)
			rm -f ./results/$tmp_file
			# salva resultado
			echo "$stm, $nb_ro_obj, $avg, $sd" # debug
			echo "$stm, $nb_ro_obj, $avg, $sd" >> ./results/$exp_file
		done
 	done
  	# grafico
  	Rscript --vanilla graph_obj.R ./results/$exp_file ./results/$exp_file
}


# Experimento variando-se o numero de threads
# $1: nome do experimento
# $2: numero de contas
# $3: numero de threads
# $4: numero de blocos
# $5: probabilidade de somente leitura
# $6: numero de objetos somente leitura
# $7: numero de objetos leitura e escrita
function exp_bank_thr(){
	local name=$1
	local nb_acc=$2
	local nb_thr=$3
	#local nb_bloc=$4
	local prob=$5
 	local nb_ro=$6
  	local nb_rw=$7
	local exp_file="exp_thr.$name.$prob.csv"
	#echo "file: $exp_file" # debug
	# cabecalho
	echo "stm, threads, throughput, sd" # debug
 	echo "stm, threads, throughput, sd" > ./results/$exp_file
	for stm in $STM; do
		for nb_bloc in $SEQ_THREADS; do
			local run_file=$name.$APP.$stm.$nb_acc.$nb_thr.$nb_bloc.$prob.$nb_ro.$nb_rw.txt
			echo $run_file # debug
			echo "" > ./results/$run_file
			for rep in $(seq 1 $NB_REPEAT); do
   				# executa aplicacao
				printf "$rep " # debug
   				./$stm/$APP $nb_acc $nb_thr $nb_bloc $prob $nb_ro $nb_rw 1 >> ./results/$run_file
			done
			echo "" # debug
  			# processamento resultados
			local tmp_file=$run_file.tmp
			cat ./results/$run_file | filter_throughtput > ./results/$tmp_file
			#cat ./results/$run_file
			cat ./results/$tmp_file
			avg=$(Rscript --vanilla mean_file.R ./results/$tmp_file)
			sd=$(Rscript --vanilla sd_file.R  ./results/$tmp_file)
			rm -f ./results/$tmp_file
			# salva resultado
			echo "$stm, $nb_bloc, $avg, $sd" # debug
			echo "$stm, $nb_bloc, $avg, $sd" >> ./results/$exp_file
		done
 	done
  	# grafico
   	Rscript --vanilla graph_thr.R ./results/$exp_file ./results/$exp_file
}


#compile

# Limpar
#rm -f ./results/*
mkdir -p results

echo "Teste 1: `date`"; exp_bank_prob "high" $NB_ACC_HIGH $NB_THREADS $NB_BLOCKS 8 $RO_SIZE $RW_SIZE
echo "Teste 2: `date`"; exp_bank_prob "low" $NB_ACC_LOW $NB_THREADS $NB_BLOCKS 8 $RO_SIZE $RW_SIZE
echo "Teste 3: `date`"; exp_bank_obj "high" $NB_ACC_HIGH $NB_THREADS $NB_BLOCKS 8 $RO_SIZE $RW_SIZE
echo "Teste 4: `date`"; exp_bank_obj "low" $NB_ACC_LOW $NB_THREADS $NB_BLOCKS 8 $RO_SIZE $RW_SIZE
echo "Teste 5: `date`"; exp_bank_obj "high" $NB_ACC_HIGH $NB_THREADS $NB_BLOCKS 2 $RO_SIZE $RW_SIZE
echo "Teste 6: `date`"; exp_bank_obj "low" $NB_ACC_LOW $NB_THREADS $NB_BLOCKS 2 $RO_SIZE $RW_SIZE

NB_THREADS=128
NB_BLOCKS=64
SEQ_THREADS="8 16 24 32 40 48 56 64"
echo "Teste 7: `date`"; exp_bank_thr "high" $NB_ACC_HIGH $NB_THREADS $NB_BLOCKS 8 $RO_SIZE $RW_SIZE
echo "Teste 8: `date`"; exp_bank_thr "low" $NB_ACC_LOW $NB_THREADS $NB_BLOCKS 8 $RO_SIZE $RW_SIZE
echo "Teste 9: `date`"; exp_bank_thr "high" $NB_ACC_HIGH $NB_THREADS $NB_BLOCKS 2 $RO_SIZE $RW_SIZE
echo "Teste 10: `date`"; exp_bank_thr "low" $NB_ACC_LOW $NB_THREADS $NB_BLOCKS 2 $RO_SIZE $RW_SIZE

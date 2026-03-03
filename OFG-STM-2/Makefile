all: bank

bank:
	nvcc -rdc=true --maxrregcount 63 bank.cu -o bank

clean:
	rm bank
	
all: bank

bank: 
	nvcc -rdc=true --maxrregcount 63  exe.cu -o bank

clean:
	rm bank
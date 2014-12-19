# Makefile
# Generic Makefile for making cuda programs
#
BIN				:= rnp32is-gpu
# flags
CUDA_INSTALL_PATH		:= /opt/cuda
OBJDIR				:= obj
INCLUDES			+= -I$(CUDA_INSTALL_PATH)/include
LIBS				:= -L$(CUDA_INSTALL_PATH)/lib64
CFLAGS				:= -O3
LDFLAGS				:= -lrt -lm -lcudart
# compilers
#NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive -arch sm_20 --ptxas-options=-v
NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive -arch sm_30 --ptxas-options=-v -use_fast_math 
#NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc -g -G --compiler-options -fpermissive -arch sm_13 --ptxas-options=-v -use_fast_math 
CC				:= g++
LINKER				:= g++ -fPIC
# files
C_SOURCES			:= io.cpp
#	dcdio.c \
	pdbio.c \
	topio.c \
	config_reader.c \
	param_initializer.c \
	sop.c 
CU_SOURCES			:= main.cu
C_OBJS				:= $(patsubst %.c, $(OBJDIR)/%.c.o, $(C_SOURCES))
CU_OBJS				:= $(patsubst %.cu, $(OBJDIR)/%.cu.o, $(CU_SOURCES))
 
$(BIN): makedirs $(CU_OBJS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_SOURCES) $(LDFLAGS) $(INCLUDES) $(LIBS)
 
$(OBJDIR)/%.c.o: $(C_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<
 
$(OBJDIR)/%.cu.o: $(CU_SOURCES)
	$(NVCC) $(INCLUDES) -o $@ -c $<
	
#gsop-top:
#	$(CC) -o sop-top soptop.c config_reader.c topio.c
 
makedirs:
	mkdir -p $(OBJDIR)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_INSTALL_PATH)/lib ./$(BIN)
 
clean:
	rm -f $(BIN) sop-top $(OBJDIR)/*.o
	
install: $(BIN) xyz2fullpdb.py pdb2rnp.py 
#	cp $(BIN) /usr/bin/$(BIN)
	cp $(BIN) /home/zhur/bin/$(BIN)
	cp pdb2rnp.py /home/zhur/bin/
#	cp xyz2fullpdb.py /home/zhur/bin/

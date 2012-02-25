# EDIT

export LD_LIBRARY_PATH=../../../culaR12/lib:$LD_LIBRARY_PATH

CULA_PATH = /home/feel/documents/informatics/programming/cuda/lib/cula/culaR12

CC = nvcc
LIB = -L ${CULA_PATH}/lib -L /usr/lib/ -ldl -lcppunit -lcula -lcublas -lcudart
INCL = -I ../ -I ${CULA_PATH}/include/ -I /usr/include/

CFLAGS_COMMON = -arch=sm_11 -Xcompiler -fmessage-length=0
CFLAGS = -O2 

SYMS = -DVERBOSE #-DVVERBOSE

OUT1 = test_cula
FILE1 = test_cula.cu


OUT_ALL = $(OUT1) $(OUT2)


all: $(OUT_ALL)

test_cula:
	$(CC) $(SYMS) $(CFLAGS_COMMON) $(CFLAGS) $(LIB) $(INCL) -o $(OUT1) $(FILE1)

clean: 
	rm $(OUT_ALL)


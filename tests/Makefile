
CULA_PATH = /home/feel/documents/informatics/programming/cuda/lib/cula/culaR12
CUSP_PATH = /opt/cuda-toolkit/include #/home/feel/documents/university/upv/works/iterative-solvers/trunk/cusp


CC = nvcc
LIB = -L ${CULA_PATH}/lib -lcula
#LIB += -L ../lib/ -lcuspla
LIB += -ldl -lcppunit -lcublas -lcudart

INCL = -I $(CUSP_PATH)
INCL += -I $(CULA_PATH)/include
INCL += -I ../include

CFLAGS_COMMON = -O2 -arch=sm_11 -Xcompiler -fmessage-length=0
CFLAGS = 

SYMS = -DVERBOSE #-DVVERBOSE

OUT1 = test_cuspla
FILE1 = test_cuspla.cu


OUT_ALL = $(OUT1)


all: $(OUT_ALL)

test_cuspla:
	$(CC) $(SYMS) $(CFLAGS_COMMON) $(CFLAGS) $(LIB) $(INCL) -o $(OUT1) $(FILE1)

clean:
	rm $(OUT_ALL)


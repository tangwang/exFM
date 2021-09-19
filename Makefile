ifndef dim
dim=15
endif 

SRC = src/feature/dense_fea.cc \
      src/feature/common_fea.cc \
      src/feature/sparse_fea.cc \
      src/feature/varlen_sparse_fea.cc \
      src/feature/fea_manager.cc \
      src/train/train_opt.cc \
      src/train/train.cc \
      src/solver/solver_factory.cc \
      src/solver/base_solver.cc \

DEPEND_INCLUDES =  ${wildcard  src/feature/*.h} \
	  ${wildcard  src/solver/*.h} \
	  ${wildcard  src/solver/ftrl/*.h} \
	  ${wildcard  src/solver/adam/*.h} \
	  ${wildcard  src/solver/sgdm/*.h} \
	  ${wildcard  src/utils/*.h} \
	  ${wildcard  third_party/*.h} \
	  ${wildcard  src/train/*.h} \


OBJ = ${patsubst %.cc, %.o, ${SRC}}
OBJ_DEBUG = ${patsubst %.cc, %.o_DEBUG, ${SRC}}

all : bin/train bin/train_debug

CC = g++ -fmax-errors=4 -DDIM=${dim}
LIB= -lpthread
INC = -I./third_party  -I./src
DEBUG_CCFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -Wno-reorder 
CCFLAGS = -g  -std=c++11 -Wall -Wno-sign-compare -Wno-reorder 
#CCFLAGS = -g -O3 -std=c++11 -Wall -Wno-sign-compare -Wno-reorder 

bin/train: ${OBJ} 
	-mkdir -p bin
	${CC} ${CCFLAGS}  ${LIB} ${OBJ} -o $@
	@echo "Compile done."

bin/train_debug: ${OBJ_DEBUG} 
	-mkdir -p bin
	${CC} ${DEBUG_CCFLAGS} ${LIB} ${OBJ_DEBUG} -o $@
	@echo "Compile DEBUG version done."

$(OBJ):%.o:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${CCFLAGS} ${INC} -c $< -o $@
	${CC} ${DEBUG_CCFLAGS} -D_DEBUG_VER_ ${INC} -c $< -o $@_DEBUG

clean:
	@rm -f ${OBJ}
	@rm -f ${OBJ_DEBUG}
	@echo "Clean object files done."

	@rm -f *~
	@echo "Clean tempreator files done."

	@rm -f bin/train
	@rm -f bin/train_debug
	@echo "Clean target files done."

	@echo "Clean done."


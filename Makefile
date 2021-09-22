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
	  ${wildcard  src/solver/adagrad/*.h} \
	  ${wildcard  src/utils/*.h} \
	  ${wildcard  third_party/*.h} \
	  ${wildcard  src/train/*.h} \


OBJS = ${patsubst %.cc, %.o, ${SRC}}
DEBUG_OBJS = ${patsubst %.cc, %.o_DEBUG, ${SRC}}

all : bin/train bin/train_debug

CC = g++
LIB= -lpthread
INC = -I./third_party  -I./src
DEBUG_CCFLAGS = -g -std=c++11 -Wall -fmax-errors=4 -DDIM=${dim}
CCFLAGS = -g -O3 -std=c++11 -Wall -fmax-errors=4 -DDIM=${dim}

bin/train: ${OBJS} 
	-mkdir -p bin
	${CC} ${LIB} ${OBJS} -o $@
	@echo "Compile done."

bin/train_debug: ${DEBUG_OBJS} 
	-mkdir -p bin
	${CC} ${LIB} ${DEBUG_OBJS} -o $@
	@echo "Compile DEBUG version done."

$(OBJS):%.o:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${CCFLAGS} ${INC} -c $< -o $@
	${CC} ${DEBUG_CCFLAGS} -D_DEBUG_VER_ ${INC} -c $< -o $@_DEBUG

clean:
	@rm -f ${OBJS}
	@rm -f ${DEBUG_OBJS}
	@echo "Clean object files done."

	@rm -f *~
	@echo "Clean tempreator files done."

	@rm -f bin/train
	@rm -f bin/train_debug
	@echo "Clean target files done."

	@echo "Clean done."


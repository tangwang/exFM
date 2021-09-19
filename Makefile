
CUR_DIR=.

DEPEND_INCLUDES =  ${wildcard  src/feature/*.h} \
	  ${wildcard  src/solver/*.h} \
	  ${wildcard  src/solver/ftrl/*.h} \
	  ${wildcard  src/solver/adam/*.h} \
	  ${wildcard  src/solver/sgdm/*.h} \
	  ${wildcard  src/utils/*.h} \
	  ${wildcard  third_party/*.h} \
	  ${wildcard  src/train/*.h} \

SRC = src/feature/dense_fea.cc \
      src/feature/common_fea.cc \
      src/feature/sparse_fea.cc \
      src/feature/varlen_sparse_fea.cc \
      src/feature/fea_manager.cc \
      src/train/train_opt.cc \
      src/train/train.cc \
      src/solver/solver_factory.cc \
      src/solver/base_solver.cc \


OBJ = ${patsubst %.cc, %.o, ${SRC}}

all : bin/train


CC = g++ -fmax-errors=4 -DDIM=8
LIB= -lpthread
INC = -I./third_party  -I./src
CCFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -Wno-reorder ${INC} 
# CCFLAGS = -g -std=c++11 -O3 -Wall ${INC}

bin/train: ${OBJ} 
	-mkdir -p bin
	${CC} ${CCFLAGS}  ${LIB} ${OBJ} -o $@
	@echo "Compile done."


$(OBJ):%.o:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${CCFLAGS} -c $< -o $@

clean:
	@rm -f ${OBJ}
	@echo "Clean object files done."

	@rm -f *~
	@echo "Clean tempreator files done."

	@rm -f bin/train
	@echo "Clean target files done."

	@echo "Clean done."


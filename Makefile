
CUR_DIR=.

DEPEND_INCLUDES =  ${wildcard  src/feature/*.h} \
	  ${wildcard  src/ftrl/*.h} \
	  ${wildcard  src/utils/*.h} \
	  ${wildcard  third_party/*.h} \
	  ${wildcard  src/train/*.h} \

SRC = src/feature/dense_fea.cc \
      src/ftrl/param_container.cc \
      src/ftrl/train_opt.cc \
      src/feature/sparse_fea.cc \
      src/feature/varlen_sparse_fea.cc \
      src/feature/fea_manager.cc \
      src/ftrl/ftrl_learner.cc \
      src/train/train.cc \

OBJ = ${patsubst %.cc, %.o, ${SRC}}

all : bin/train

CC = g++
LIB= -lpthread
INC = -I./third_party  -I./src
CCFLAGS = -g -std=c++11 -O3 -Wall ${INC}

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


ifndef dim
dim=15
endif 

SRC = src/feature/dense_feat.cc \
      src/feature/sparse_feat.cc \
      src/feature/varlen_sparse_feat.cc \
      src/feature/feat_manager.cc \
      src/train/train_opt.cc \
      src/solver/solver_factory.cc \
      src/solver/base_solver.cc \
	  third_party/murmur_hash3/MurmurHash3.cc 

SRC_TRAIN = $(SRC) src/train/train.cc 

SRC_PRED = $(SRC) src/train/predict.cc 

SRC_PRED_LIB = $(SRC) src/train/lib_fm_pred.cc 

DEPEND_INCLUDES =  ${wildcard  src/feature/*.h} \
	  ${wildcard  src/solver/*.h} \
	  ${wildcard  src/solver/ftrl/*.h} \
	  ${wildcard  src/solver/adam/*.h} \
	  ${wildcard  src/solver/sgdm/*.h} \
	  ${wildcard  src/solver/adagrad/*.h} \
	  ${wildcard  src/utils/*.h} \
	  ${wildcard  third_party/*.h} \
	  ${wildcard  src/train/*.h} \

OBJS_TRAIN = ${patsubst %.cc, %.o, ${SRC_TRAIN}}
OBJS_PRED = ${patsubst %.cc, %.o, ${SRC_PRED}}
OBJS_PRED_LIB = ${patsubst %.cc, %.sharedO, ${SRC_PRED_LIB}}
DEBUG_OBJS_TRAIN = ${patsubst %.cc, %.debugO, ${SRC_TRAIN}}

# all : bin/train bin/train_debug lib/fm_pred.so bin/pred
all : bin/train  

CC = g++
LIB= -lpthread
INC = -I./third_party  -I./src
DEBUG_CCFLAGS = -g -O0 -fno-inline -std=c++11 -Wall -fmax-errors=4 -DDIM=${dim} -Wno-unused-local-typedefs -Wno-attributes
LIB_CCFLAGS =  -fPIC -shared -O3 -funroll-loops -std=c++11 -Wall -fmax-errors=4 -DDIM=${dim} -Wno-unused-local-typedefs -Wno-attributes -march=native 
CCFLAGS = -g -O3 -funroll-loops -std=c++11 -Wall -fmax-errors=4 -DDIM=${dim} -Wno-unused-local-typedefs -Wno-attributes -march=native 

lib/fm_pred.so: ${OBJS_PRED_LIB} 
	-mkdir -p lib
	${CC} -fPIC -shared ${LIB} ${OBJS_PRED_LIB} -o $@

bin/train: ${OBJS_TRAIN} 
	-mkdir -p bin
	${CC} ${LIB} ${OBJS_TRAIN} -o $@
	@echo "Compile done."

bin/pred: ${OBJS_PRED} 
	-mkdir -p bin
	${CC} ${LIB} ${OBJS_PRED} -o $@
	@echo "Compile done."

bin/train_debug: ${DEBUG_OBJS_TRAIN} 
	-mkdir -p bin
	${CC} ${LIB} ${DEBUG_OBJS_TRAIN} -o $@
	@echo "Compile DEBUG version done."

$(OBJS_TRAIN):%.o:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${CCFLAGS} ${INC} -c $< -o $@

$(DEBUG_OBJS_TRAIN):%.debugO:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${DEBUG_CCFLAGS} -D_DEBUG_VER_ ${INC} -c $< -o $@

$(OBJS_PRED_LIB):%.sharedO:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${LIB_CCFLAGS} ${INC} -c $< -o $@

$(OBJS_PRED):%.o:%.cc ${DEPEND_INCLUDES}
	@echo "Compiling $< ==> $@"
	${CC} ${CCFLAGS} ${INC} -c $< -o $@

clean:
	@rm -f ${OBJS_TRAIN}
	@rm -f ${DEBUG_OBJS_TRAIN}
	@rm -f ${OBJS_PRED}
	@rm -f ${OBJS_PRED_LIB}
	@echo "Clean object files done."

	@rm -f *~
	@echo "Clean tempreator files done."

	@rm -f bin/train
	@rm -f bin/train_debug
	@echo "Clean target files done."

	@echo "Clean done."


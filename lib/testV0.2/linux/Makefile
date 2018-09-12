CAFFE2_PREFIX=/root/pytorch/build
CXXFLAGS = `pkg-config --cflags opencv`  -I${CAFFE2_PREFIX}  -w -I/usr/local/cuda/include -std=c++11 
LIBS=`pkg-config --libs opencv`  -L${CAFFE2_PREFIX}/lib  -L/usr/local/cuda/lib64 -lcaffe2_gpu -lcaffe2  -lglog  -lcudart -lcurand
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

$(warning  $(OBJS))


predictor: $(OBJS)
	$(CXX)  $^ $(LIBS) -o ${@}

clean:
	$(RM) predictor $(OBJS)

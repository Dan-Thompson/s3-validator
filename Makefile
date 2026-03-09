CXX := g++
CXXFLAGS_MULTI := -std=c++17 -O3 -pthread
CXXFLAGS_SINGLE := -std=c++17 -O2

MULTI_SRC := crc64nvme_s3-multicore.cpp
SINGLE_SRC := crc64nvme_s3-singlecore.cpp

MULTI_BIN := crc64nvme_s3_multi
SINGLE_BIN := crc64nvme_s3

.PHONY: all clean test

all: $(MULTI_BIN) $(SINGLE_BIN)

$(MULTI_BIN): $(MULTI_SRC)
	$(CXX) $(CXXFLAGS_MULTI) $< -o $@

$(SINGLE_BIN): $(SINGLE_SRC)
	$(CXX) $(CXXFLAGS_SINGLE) $< -o $@

test: all
	./$(MULTI_BIN) --self-test
	./$(SINGLE_BIN) --self-test

clean:
	rm -f $(MULTI_BIN) $(SINGLE_BIN)

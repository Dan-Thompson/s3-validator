CXX ?= c++
CXXFLAGS ?= -O3
PYTHON ?= python3

MULTI_SRC := crc64nvme_s3-multicore.cpp

BUILD_DIR ?= build
MULTI_BIN := $(BUILD_DIR)/crc64nvme_s3_multi
PORTABLE_BIN := $(BUILD_DIR)/crc64nvme_s3_multi_portable
TEST_BIN := $(BUILD_DIR)/test_crc64nvme
PORTABLE_TEST_BIN := $(BUILD_DIR)/test_crc64nvme_portable

.PHONY: all portable clean test

all: $(MULTI_BIN)

portable: $(PORTABLE_BIN)

$(MULTI_BIN): $(MULTI_SRC) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -pthread $< $(LDFLAGS) $(LDLIBS) -o $@

$(PORTABLE_BIN): $(MULTI_SRC) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -pthread -DCRC64NVME_PORTABLE $< $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_BIN): tests/test_crc64nvme.cpp $(MULTI_SRC) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -pthread $< $(LDFLAGS) $(LDLIBS) -o $@

$(PORTABLE_TEST_BIN): tests/test_crc64nvme.cpp $(MULTI_SRC) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -pthread -DCRC64NVME_PORTABLE $< $(LDFLAGS) $(LDLIBS) -o $@

test: all $(TEST_BIN) $(PORTABLE_TEST_BIN)
	$(abspath $(MULTI_BIN)) --self-test
	$(abspath $(TEST_BIN))
	$(abspath $(PORTABLE_TEST_BIN))
	$(PYTHON) tests/test_cli.py $(MULTI_BIN)

clean:
	rm -rf $(BUILD_DIR)

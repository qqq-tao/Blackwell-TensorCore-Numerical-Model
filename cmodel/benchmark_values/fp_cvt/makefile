BASENAME := fpemu
PYTHON ?= python3
NVCC ?= nvcc
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')
CXXFLAGS ?= -O3 --compiler-options -fPIC -shared -std=c++17

all: $(BASENAME)$(EXT_SUFFIX)

$(BASENAME)$(EXT_SUFFIX): $(BASENAME).cpp
	$(NVCC) $(CXXFLAGS) $(PYBIND11_INCLUDES) $< -o $@

.PHONY: clean
clean:
	rm -f $(BASENAME).so $(BASENAME).cpython-*.so

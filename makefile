# MakeFile

# Python Scripts
PYMESH = python/mesh.py
PYDADOS = python/data_process.py

# Program Name
EXE = lbm

# Compilers
NVCC := nvcc

LIBRARY := yaml-cpp sundials_kinsol sundials_nvecserial sundials_nvecmanyvector
LIBDIR := /usr/local/lib

# GPU
ARCH = sm_61

# Directories
SRCDIR := src
INCDIR := inc /usr/local/include
OBJDIR := obj
BINDIR := bin
VELOCITY := bin/Velocity
RESULTS := bin/Results
MESH := bin/Mesh

# Files
C_FILES := $(wildcard $(SRCDIR)/*.c)
CU_FILES := $(wildcard $(SRCDIR)/*.cu)
CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)

C_OBJ := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(C_FILES))
CU_OBJ := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CU_FILES))
CPP_OBJ := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPP_FILES))

C_DEP := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.d, $(C_FILES))
CU_DEP := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.d, $(CU_FILES))
CPP_DEP := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.d, $(CPP_FILES))

OBJ := $(C_OBJ) $(CPP_OBJ) $(CU_OBJ)
DEP := $(C_DEP) $(CPP_DEP) $(CU_DEP)

# Flags
CFLAGS := -O3 -DNDEBUG
NVCCARCHFLAG := -arch sm_61
NVCCFLAGS := -std=c++11 -v --ptxas-options=-v -O3 --device-c # Para debug flag -g -G e executar com cuda-memcheck ./lbm |more
LDFLAGS := $(addprefix -L, $(LIBDIR))
DEPFLAGS := -MMD
LINKFLAGS := -Xcompiler \"-Wl,-rpath,/usr/local/lib\"

INCLUDES := $(addprefix -I, $(INCDIR))
LIBRARIES := $(addprefix -l, $(LIBRARY)) -lm /usr/lib/x86_64-linux-gnu/librt.so

ALL_CFLAGS := $(DEPFLAGS)
ALL_CFLAGS += $(CFLAGS)

ALL_CPPFLAGS := $(DEPFLAGS)
ALL_CPPFLAGS += $(NVCCARCHFLAG)
ALL_CPPFLAGS += $(NVCCFLAGS)

ALL_LDFLAGS := $(LDFLAGS)
ALL_LDFLAGS += $(LINKFLAGS)

CXXFLAGS := -std=c++11
NVCCARCHFLAG := -arch $(ARCH)
NVCCFLAGS := -v --ptxas-options=-v -O3 # Para debug flag -g -G e executar com cuda-memcheck ./lbm |more

COMPILE.c = $(NVCC) $(ALL_CFLAGS) -g $(INCLUDES) -c
COMPILE.cpp = $(NVCC) $(ALL_CPPFLAGS) $(INCLUDES) --device-c
COMPILE.cu = $(NVCC) $(ALL_CPPFLAGS) $(INCLUDES) --device-c

.PHONY: all clean mesh plot refresh run

all: $(EXE)

# Linkage
$(EXE): $(OBJ)
	$(NVCC) $(NVCCARCHFLAG) $(INCLUDES) $(ALL_LDFLAGS) $^ -o $@ $(LIBRARIES)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(COMPILE.c) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(COMPILE.cpp) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(COMPILE.cu) $< -o $@

$(OBJDIR):
	@mkdir -p $@

# Cleaning up
clean:
	@echo Cleaning up...
	@rm -f -r $(OBJDIR)
	@rm -f -r $(VELOCITY)/*
	@rm -f -r $(RESULTS)/*
	@rm -f -r $(MESH)/*
	@rm -f $(EXE)
	@rm -f $(BINDIR)/*.gif
	@rm -f $(BINDIR)/*.mp4
	@echo Done!

mesh:
	@echo Generating mesh...
	@python $(PYMESH)
	@echo Done!

plot:
	@python $(PYDADOS)

refresh:
	@echo Cleaning up the images...
	@rm -f -r $(VELOCITY)/*
	@echo Cleaning up the results...
	@rm -f -r $(RESULTS)/*
	@rm -f $(BINDIR)/*.gif
	@rm -f $(BINDIR)/*.mp4
	@echo Cleaning up the mesh...
	@rm -f -r $(MESH)/*
	@echo Done!

# Running
run:
	@echo Generating mesh...
	@python $(PYMESH)
	
	@echo Running simulation...
	@./$(EXE)

	@echo Running Python Script...
	@python $(PYDADOS)

# Including dependency
-include $(DEP)
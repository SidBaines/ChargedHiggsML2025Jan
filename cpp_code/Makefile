# Directories
SDIR    = ./
IDIR    = ./
LDIR    = ./
LWTNN   = ../../ChargedHiggs_CodeForThesis/lwtnn
BUILD   = $(LWTNN)/build/lib/liblwtnn-stat.a

# Compiler and flags
CXX     = g++
CCFLAGS = -O2 -g  # Add specific flags if needed

# ROOT configurations
ROOT_CFLAGS = $(shell root-config --cflags)
ROOT_LIBS    = $(shell root-config --libs)
LIBS          = $(ROOT_LIBS) -lTMVA -lMLP -lTreePlayer -lMinuit
INCLUDE       = $(ROOT_CFLAGS) -I$(LWTNN)/include

# Source files
SOURCES = main.C main/EventLoop.C utilis/NeutrinoBuilder.C
HEADERS = utilis/configparser.h

# Output binary
BINS = bin/roo
BINS2 = bin/extra

# Default target
all: $(BINS) $(BINS2)

# Build rule for the binary
$(BINS): %: $(SOURCES) $(HEADERS)
	@echo "Building $@ ... "
	$(CXX) $(CCFLAGS) $^ -I$(IDIR) $(INCLUDE) $(LIBS) $(BUILD) -o $@
	@echo "Done"

$(BINS2): % : GetNumEntriesMain.C
	@echo -n "Building $@ ... "
	$(CXX) $(CCFLAGS) -g $< -I$(IDIR) $(INCLUDE) $(LIBS) -o getNumEntriesExecute
	@echo "Done"

# Clean rule (optional)
clean:
#	rm -f bin/roo bin/getNumEntriesExecute
	rm bin/roo bin/getNumEntriesExecute

.PHONY: all clean

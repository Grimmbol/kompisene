# Every png file needs to have a matching obj file in the input folder
OUTPUTS := output/sphere.png
OUTPUTS += output/prop.png
OUTPUTS += output/head.png
OUTPUTS += output/scene.png

WIDTH := 1920
HEIGHT := 1080

OPTIMIZATION := 0

GOLDEN := 69f9a8e630f647c086f3643a6c15e468  output/sphere.png
GOLDEN += db540503f7346bd713fe5aca3a4e7603  output/prop.png
GOLDEN += 85dac442d8c1a2bb24a9d14b50b9157b  output/head.png
GOLDEN += f7ab5ab456b5e72079817f8cfb93d055  output/scene.png

ifdef INPUT
ifndef OUTPUT
OUTPUT := output/$(patsubst %.obj,%.png,$(notdir $(INPUT)))
endif
endif

ifndef OUTPUT
OUTPUT := $(firstword $(OUTPUTS))
endif

ifndef INPUT
INPUT  := input/$(patsubst %.png,%.obj,$(notdir $(OUTPUT)))
endif

ARGUMENTS := -i $(INPUT) -o $(OUTPUT) -w $(WIDTH) -h $(HEIGHT)

SOURCE_DIR := src
BUILD_DIR  := cpurender
BINARY := $(BUILD_DIR)/cpurender

CXX := g++
FLAGS :=
CXXFLAGS := -Wall -Wextra -Wpedantic -std=c++11 -O$(OPTIMIZATION) $(FLAGS)

CHECK_FLAGS := $(BUILD_DIR)/.flags_$(shell echo '$(CXXFLAGS) $(DEFINES)' | md5sum | awk '{print $$1}')

SOURCE_PATHS := $(shell find $(SOURCE_DIR) -type f -name '*.cpp')
INCLUDE_DIRS := $(shell find $(SOURCE_DIR) -type f -name '*.hpp' -exec dirname {} \; | uniq)
OBJECTS     := $(SOURCE_PATHS:%.cpp=%.o)

INCLUDES := $(addprefix -I,$(INCLUDE_DIRS))

.PHONY: all verify call $(OUTPUTS)

help:
	@echo "TDT4200 Assignment 1"
	@echo ""
	@echo "Targets:"
	@echo "	all 		Builds $(BINARY)"
	@echo "	call		executes $(BINARY)"
	@echo "	verify		renders $(INPUT) verifies golden standard"
	@echo "	sse   		executes sse test"
	@echo "	clean		cleans up everything"
	@echo ""
	@echo "Render Targets:"
	@echo "\t$(OUTPUTS)" | sed 's/ /\n\t/g'
	@echo ""
	@echo "Options:"
	@echo "	FLAGS=$(FLAGS)"
	@echo "	OPTIMIZATION=$(OPTIMIZATION)"
	@echo "	WIDTH=$(WIDTH)"
	@echo "	HEIGHT=$(HEIGHT)"
	@echo "	INPUT=$(INPUT)"
	@echo "	OUTPUT=$(OUTPUT)"
	@echo "	PROFILE=$(PROFILE)"
	@echo "	DEFINES=$(DEFINES)"
	@echo ""
	@echo "Compiler Call:"
	@echo "	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c dummy.cpp -o dummy.o"
	@echo "Binary Call:"
	@echo "	$(PROFILE) $(BINARY) $(ARGUMENTS)"

all:
	@$(MAKE) --no-print-directory $(BINARY)


call: $(BINARY)
	$(PROFILE) $(BINARY) $(ARGUMENTS)

verify:
	@$(MAKE) --no-print-directory WIDTH=1920 HEIGHT=1080 $(OUTPUT)
	@echo "$(GOLDEN)" | grep -q "$$(md5sum $(OUTPUT))" && \
		echo "$(OUTPUT) verification successful!" || \
		echo "$(OUTPUT) file's MD5 does not match!"; \

sse: ARGUMENTS = --sse -i $(INPUT)
sse:
	$(MAKE) ARGUMENTS="$(ARGUMENTS)" call

clean:
	rm -f $(BUILD_DIR)/.flags_*
	rm -f $(BINARY)
	rm -f $(OBJECTS)

$(OUTPUTS): $(BINARY)
	@$(MAKE) --no-print-directory INPUT=input/$(patsubst %.png,%.obj,$(notdir $@)) OUTPUT=$@  call

$(OBJECTS) : %.o : %.cpp $(CHECK_FLAGS)
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

$(CHECK_FLAGS):
	@$(MAKE) --no-print-directory clean
	@mkdir -p $(dir $@)
	@touch $@

$(BINARY): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LINKING) -o $@


MAKE			+= --silent
CC      		:= gcc
INCLUDE 		:= ../include
BUILD_DIR   	:= ./build
BIN_DIR     	:= ./bin
SRC				:= ../src

CFLAGS	  	:= -I$(INCLUDE) -Wall -O3 -Werror -std=c11 -fPIC
LIB_FLAGS	:= -lm -fopenmp --shared

LIB_OBJ = libknngraph.so

ifeq ($(DEBUG), ON)
  CFLAGS += -g3
endif

OBJS := $(patsubst $(SRC)/%.c,%.o, $(wildcard $(SRC)/*.c))

OBJ  := $(patsubst %,$(BUILD_DIR)/%,$(OBJS))

$(BIN_DIR)/$(LIB_OBJ): $(OBJ)
	@$(MAKE) bin_dir
	@$(CC) -o $@ $^ $(LIB_FLAGS)

$(BUILD_DIR)/%.o: $(SRC)/%.c
	@$(MAKE) build_dir
	@$(CC) $(CFLAGS) -c $< -o $@ $(LIB_FLAGS)
 
.PHONY: clean

clean:
	@rm -rf $(BIN_DIR) $(BUILD_DIR)

bin_dir:  
	@if [ ! -d $(BIN_DIR) ]; then \
		mkdir -p $(BIN_DIR); \
	fi

build_dir:  
	@if [ ! -d $(BUILD_DIR) ]; then \
		mkdir -p $(BUILD_DIR); \
	fi
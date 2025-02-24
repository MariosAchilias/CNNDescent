# ~ make [graph_test, modules_test, all]
# The executable is stored in the ./build folder and the object files in the ./bin folder 
# make all creates the executable for both tests
# Makefile options: run, valgrind, time, gdb, clean

MAKE		+= --silent
CC      	:= gcc
INCLUDE 	:= ../include
BUILD_DIR   := ../build
BIN_DIR     := ../bin
SRC			:= ../src


CFLAGS	  	:= -I$(INCLUDE) -Wall -Werror -g3 -std=c11
VAL_FLAGS 	:= valgrind  --leak-check=full --show-leak-kinds=all --track-origins=yes
LIB_FLAGS	:= -lm -fopenmp

ifeq ($(DEBUG), ON)
  CFLAGS += -g3
endif


EXEC1 := modules_test
OBJS1 := modules.o $(patsubst $(SRC)/%.c,%.o, $(wildcard $(SRC)/*.c))
OBJ1  := $(patsubst %,$(BIN_DIR)/%,$(OBJS1))


EXEC2 := graph_test
OBJS2 := graph.o $(patsubst $(SRC)/%.c,%.o, $(wildcard $(SRC)/*.c))
OBJ2  := $(patsubst %,$(BIN_DIR)/%,$(OBJS2))

EXEC3 := util_test
OBJS3 := utilities.o $(patsubst $(SRC)/%.c,%.o, $(wildcard $(SRC)/*.c))
OBJ3  := $(patsubst %,$(BIN_DIR)/%,$(OBJS3))


all: $(EXEC1) $(EXEC2) $(EXEC3)



$(EXEC1): $(OBJ1)
	@$(MAKE) build_dir
	@$(CC) -o $(BUILD_DIR)/$@ $^ $(LIB_FLAGS)


$(EXEC2): $(OBJ2)
	@$(MAKE) build_dir
	@$(CC) -o $(BUILD_DIR)/$@ $^ $(LIB_FLAGS)

$(EXEC3): $(OBJ3)
	@$(MAKE) build_dir
	@$(CC) -o $(BUILD_DIR)/$@ $^ $(LIB_FLAGS)


$(BIN_DIR)/%.o: %.c
	@$(MAKE) bin_dir
	@$(CC) $(CFLAGS) -c  $< -o $@ $(LIB_FLAGS)


$(BIN_DIR)/%.o: $(SRC)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@ $(LIB_FLAGS)



.PHONY: run clean valgrind gdb time



run:
	@for exec in $(shell find $(BUILD_DIR) -type f -executable); do \
		./$$exec; \
	done

valgrind:
	@for exec in $(shell find $(BUILD_DIR) -type f -executable); do \
		$(VAL_FLAGS) ./$$exec; \
	done

gdb:
	@for exec in $(shell find $(BUILD_DIR) -type f -executable); do \
		gdb ./$$exec; \
	done


time:
	@for exec in $(shell find $(BUILD_DIR) -type f -executable); do \
		time ./$$exec; \
	done


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
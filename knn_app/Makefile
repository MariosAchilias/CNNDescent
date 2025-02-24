# ~ make
# The executable is stored in the ./build folder and the object files in the ./bin folder 
# Makefile options: run, valgrind, time, gdb, clean
# For passing arguments you either have to change the ARGS variable manually through the Makefile or set it through the tty
# eg: make ARGS="../datasets/50k.RectNode.normal.bin" (run, valgrind, gdb, time)



MAKE			+= --silent
CC      		:= gcc
INCLUDE 		:= ../include
BUILD_DIR   	:= ../build
BIN_DIR     	:= ../bin
SRC				:= ../src
DATASETS		:= ../datasets
SOLUTIONS_DIR	:= ../solutions

ARGS = 

CFLAGS	  	:= -I$(INCLUDE) -Wall -O3 -Werror -std=c11
VAL_FLAGS 	:= valgrind  --leak-check=full --show-leak-kinds=all --track-origins=yes
LIB_FLAGS	:= -lm -fopenmp

ifeq ($(DEBUG), ON)
  CFLAGS += -g3
endif


EXEC := knn_app
OBJS := main.o $(patsubst $(SRC)/%.c,%.o, $(wildcard $(SRC)/*.c))
OBJ  := $(patsubst %,$(BIN_DIR)/%,$(OBJS))




$(BUILD_DIR)/$(EXEC): $(OBJ)
	@$(MAKE) build_dir
	@$(MAKE) solutions_dir
	@$(CC) -o $@ $^ $(LIB_FLAGS)



$(BIN_DIR)/%.o: %.c
	@$(MAKE) bin_dir
	@$(CC) $(CFLAGS) -c  $< -o $@ $(LIB_FLAGS)


$(BIN_DIR)/%.o: $(SRC)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@ $(LIB_FLAGS)



.PHONY: run clean valgrind gdb time

run:
	@./$(BUILD_DIR)/$(EXEC) $(ARGS)

valgrind:
	@$(VAL_FLAGS) ./$(BUILD_DIR)/$(EXEC) $(ARGS)

time:
	@time ./$(BUILD_DIR)/$(EXEC) $(ARGS)

gdb:
	@gdb ./$(BUILD_DIR)/$(EXEC) $(ARGS)

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

solutions_dir:  
	@if [ ! -d $(SOLUTIONS_DIR) ]; then \
		mkdir -p $(SOLUTIONS_DIR); \
	fi

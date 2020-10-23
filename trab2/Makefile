CC := mpicc

CFLAGS := 

LNFLAG := 

SRCFILES := $(wildcard src/*.c)

all: $(SRCFILES:src/%.c=obj/%.o)
	$(CC) $(CFLAGS) obj/*.o $(LNFLAG) -o bin/saida

obj/%.o : src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf obj/*
	rm -rf bin/*

run:
	mpirun -host 164.41.127.81 -np $(np) bin/saida -do-not-resolve

action:
	echo argument is $(argument)


# HostList
# gateway
# chococino 164.41.127.81
# cm2 164.41.127.82
# cm3 164.41.127.83
# cm4164.41.127.84

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
	mpirun -host 164.41.127.83 -np $(np) bin/saida

action:
	echo argument is $(argument)


# HostList
# gateway
# localhost
# cm2
# cm3
# cm4

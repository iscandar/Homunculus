CCFLAGS = -w -O2 -g
LFLAGS = -lm


all: and or xor minus_plus glass
	make clean


and: and.o homunculus.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

or: or.o homunculus.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

xor: xor.o homunculus.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

minus_plus: minus_plus.o homunculus.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

glass: glass.o homunculus.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

.c.o:
	$(CC) -c $(CCFLAGS) $< -o $@


clean:
	rm -f *.o
	echo "Clean done"

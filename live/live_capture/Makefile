CC=gcc
CFLAGS=-g
LINKS = -lpcap
DEPS = filtering.h faster.h
OBJ = faster.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(LINKS)

faster: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LINKS)
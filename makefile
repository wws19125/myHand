all: handDetection
#compiler
CC = g++
#include
#INCLUDE = .
#Cflags
CONFIG = `pkg-config --cflags opencv --libs opencv`
handDetection: MyHandle 
MyHandle: MyHandle.cpp
	$(CC) $(CONFIG)  MyHandle.cpp -o hand
#handleMethod.o: handleMethod.cpp handleClass.h
#	$(CC) -c handleMethod.cpp

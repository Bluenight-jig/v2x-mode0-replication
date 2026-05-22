# Makefile for the V2X channel-simulation bridge
# Standalone build - no NS-3 dependency required

CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall
LDFLAGS = -lzmq -ljsoncpp

TARGET = v2x_bridge
SRC = v2x_bridge.cc

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean

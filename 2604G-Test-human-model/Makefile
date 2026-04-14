.PHONY: all configure build rebuild clean run shaders

CONFIG_PRESET=linux-default
BUILD_PRESET=linux-default
TARGET=Test

all: build

configure:
	cmake --preset $(CONFIG_PRESET)

build: configure
	cmake --build --preset $(BUILD_PRESET) --target $(TARGET) -j

debug:
	cmake --preset linux-debug
	cmake --build --preset linux-debug --target $(TARGET) -j

rebuild: clean build

clean:
	cmake --build --preset $(BUILD_PRESET) --target clean

run: build
	/home/roots/develop/build/Test/$(TARGET)

shaders:
	glslc -g -O0 shaders/shader.vert -o shaders/shader.vert.spv
	glslc -g -O0 shaders/shader.frag -o shaders/shader.frag.spv
	glslc -g -O0 shaders/ruler.vert -o shaders/ruler.vert.spv
	glslc -g -O0 shaders/ruler.frag -o shaders/ruler.frag.spv

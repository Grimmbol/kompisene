# TDT4200-Assignment-1

### Build Instructions

1. Using make (recommended)
    ```bash
    make
    make all
    ```

2. Using cmake (replaces Makefile!)
    ```bash
    cmake .
    make
    ```

3. manual compiling
    ```bash
    g++ -Wall -Wextra -Wpedantic -std=c++11 -O3 -Isrc/utilities -Isrc -c src/utilities/lodepng.cpp -o src/utilities/lodepng.o
    g++ -Wall -Wextra -Wpedantic -std=c++11 -O3 -Isrc/utilities -Isrc -c src/utilities/OBJLoader.cpp -o src/utilities/OBJLoader.o
    g++ -Wall -Wextra -Wpedantic -std=c++11 -O3 -Isrc/utilities -Isrc -c src/utilities/geom.cpp -o src/utilities/geom.o
    g++ -Wall -Wextra -Wpedantic -std=c++11 -O3 -Isrc/utilities -Isrc -c src/main.cpp -o src/main.o
    g++ -Wall -Wextra -Wpedantic -std=c++11 -O3 -Isrc/utilities -Isrc -c src/rasteriser.cpp -o src/rasteriser.o
    g++ -Wall -Wextra -Wpedantic -std=c++11 -O3  src/utilities/lodepng.o src/utilities/OBJLoader.o src/utilities/geom.o src/main.o src/rasteriser.o  -o cpurender/cpurender
    ```

### Calling application

1. Using make
    ```bash
    make call
    ```

2. manual
    ```bash
    cpurender/cpurender -i input/sphere.obj -o output/sphere.png -w 1920 -h 1080
    ```

### Switching render target

1. Using make
    ```bash
    make call OUTPUT=output/prop.png
    (or) make call INPUT=input/prop.obj
    ```

2. manual
    ```bash
    cpurender/cpurender -i input/prop.obj -o output/prop.png -w 1920 -h 1080
    ```
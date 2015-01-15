all: cython

cython:
	python2 setup.py build_ext --inplace
	cp *.so ../lib/cpp_wrapper

debug:
	python setup.py debug build_ext --inplace

clean:
	rm -f ../lib/cpp_wrapper/*.so
	rm -rf *.so
	rm -f *.cpp
	rm -f *.c
	rm -rf build
	rm -rf */*.o

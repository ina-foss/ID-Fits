all: cython

cython:
	python2 setup.py build_ext --inplace
	cp *.so ../lib/cpp_wrapper

complete:
	python2 setup.py build_ext --inplace use_csiro_alignment build_alignment_training
	cp *.so ../lib/cpp_wrapper

debug:
	python2 setup.py debug build_ext --inplace
	cp *.so ../lib/cpp_wrapper

clean:
	rm -f ../lib/cpp_wrapper/*.so
	rm -rf *.so
	rm -f *.cpp
	rm -f *.c
	rm -rf build
	rm -rf */*.o

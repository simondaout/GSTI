python -m cProfile -o profile:out optimize.py syntheticinput
pyprof2calltree -i profile\:out -k 

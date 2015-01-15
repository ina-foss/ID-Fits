import sys
import os

sys.path[0] = '/'.join(os.getcwd().split('/')[:-1]) + "/lib"
print sys.path[0]
os.environ['PYTHONPATH'] = ':'.join(sys.path)

import os
import imp


imp.load_source("fix_imports", os.path.join(os.path.dirname(__file__), os.pardir, "fix_imports.py"))


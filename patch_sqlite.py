# patch_sqlite.py
import sys

try:
    # force use of modern sqlite from pysqlite3-binary
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # if package isn't available, just continue
    pass
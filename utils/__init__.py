from __future__ import absolute_import, division, print_function, with_statement

import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def temp_dir_ctx():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)

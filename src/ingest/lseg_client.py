from contextlib import contextmanager
import lseg.data as ld


@contextmanager
def lseg_session():
    ld.open_session()
    try:
        yield ld
    finally:
        ld.close_session()
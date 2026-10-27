import os
from contextlib import contextmanager

import lseg.data as ld
from dotenv import load_dotenv

load_dotenv()


@contextmanager
def lseg_session():
    """[ING-01] open an LSEG Workspace session (LSEG_APP_KEY from .env, else lseg-data.config.json), always closes it"""
    app_key = os.getenv("LSEG_APP_KEY")
    if app_key:
        ld.open_session(name="desktop.workspace", app_key=app_key)
    else:
        ld.open_session()
    try:
        yield ld
    finally:
        ld.close_session()

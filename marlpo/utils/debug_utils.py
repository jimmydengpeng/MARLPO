
from rich import inspect
import rich
rich.get_console().width -= 30


def log(title: str = None, obj=None, docs=False):
    inspect(obj=obj, title=title, docs=docs)


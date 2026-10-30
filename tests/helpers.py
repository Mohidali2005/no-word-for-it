import ast

from nwf.paths import AUTHORS_SHAPES


def authors_functions(script,names,namespace):
    """Load some functions from an authors probe script without running the rest of it"""
    source = (AUTHORS_SHAPES/script).read_text(encoding="utf-8")
    keep = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=keep,type_ignores=[]),script,"exec"),namespace)
    return namespace

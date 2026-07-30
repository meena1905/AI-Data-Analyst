from __future__ import annotations
import ast
import multiprocessing as mp
import numpy as np
import pandas as pd
from app.utils.errors import QueryExecutionError
_ALLOWED_BUILTINS = {
    "len", "range", "min", "max", "sum", "sorted", "abs", "round",
    "list", "dict", "set", "tuple", "enumerate", "zip", "str", "int",
    "float", "bool", "True", "False", "None",
}
_FORBIDDEN_NODE_TYPES = (ast.Import, ast.ImportFrom)
def _static_check(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise QueryExecutionError(f"Generated pandas code has a syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise QueryExecutionError("Generated code may not import modules.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise QueryExecutionError("Generated code may not access dunder attributes.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "open", "__import__", "compile", "input"}:
                raise QueryExecutionError(f"Generated code may not call '{node.func.id}'.")
def _worker(code: str, frames: dict[str, pd.DataFrame], queue: mp.Queue) -> None:
    try:
        safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                          for k in _ALLOWED_BUILTINS if (
                              k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k)
                          )}
        namespace: dict = {"pd": pd, "np": np, **frames, "__builtins__": safe_builtins}
        exec(code, namespace)  
        result = namespace.get("result")
        if isinstance(result, pd.DataFrame):
            queue.put(("ok", result.head(1000)))
        elif isinstance(result, pd.Series):
            queue.put(("ok", result.head(1000).to_frame()))
        else:
            queue.put(("ok", pd.DataFrame({"result": [result]})))
    except Exception as e:  # noqa: BLE001
        queue.put(("error", str(e)))
def run_pandas_code(code: str, frames: dict[str, pd.DataFrame], timeout_seconds: int = 10) -> pd.DataFrame:
    _static_check(code)
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, frames, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise QueryExecutionError(f"Pandas code timed out after {timeout_seconds}s.")
    if queue.empty():
        raise QueryExecutionError("Pandas code did not produce a result (process crashed).")
    status, payload = queue.get()
    if status == "error":
        raise QueryExecutionError(f"Pandas code raised an error: {payload}")
    return payload

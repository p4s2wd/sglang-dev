"""Static rule-compliance audit for the sglang repo (stdlib only, read-only).

Checks mirror .claude/rules/*.md:
  general-code-style.md  -> file <2k LOC, function <100 LOC, no Mixin classes
  no-dataclasses.md      -> @dataclass usage (grandfathered, new code should use msgspec.Struct)
  no-getattr-defensive.md-> getattr(obj,"f",default) / hasattr(obj,"f")
  comment-style.md       -> no bare TODO, no FIXME/XXX/HACK, ASCII-only comments,
                            no Args:/Returns: docstring blocks on private helpers
  schedule-batch-out-of-place-mutation.md -> in-place mutation of ScheduleBatch fields
"""
import ast
import collections
import io
import pathlib
import re
import sys
import tokenize

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "python")

stats = collections.Counter()
big_files = []
big_funcs = []
dataclass_files = collections.Counter()
getattr_sites = []
hasattr_sites = []
bare_todo = []
bad_tags = []
nonascii_comments = collections.Counter()
private_args_docs = []
mixins = []
inplace_sched = []
self_attrs = collections.Counter()

CJK = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff01-\uffff]")
FANCY = re.compile(r"[\u2190-\u21ff\u2200-\u22ff\u25a0-\u25ff\u2b00-\u2bff\u2700-\u27bf\u00d7\u2212\u2264\u2265\u2260\u2261\u2262\u221a\u2717\u2713\u21d2\u2192\u2190\u2794\u27a1\u2795]")


def comments_of(text):
    try:
        for tok in tokenize.generate_tokens(io.BytesIO(text.encode("utf-8", "replace")).readline):
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string
    except Exception:
        return


def docstring_of(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0].value.value
    return None


def has_dataclass_decorator(node):
    for d in node.decorator_list:
        s = ast.dump(d)
        if "dataclass" in s or "'attrs'" in s or '"attrs"' in s:
            return True
    return False


def check_file(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    nloc = text.count("\n") + 1
    stats["files"] += 1
    stats["loc"] += nloc
    if nloc > 2000:
        big_files.append((nloc, str(path)))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        stats["syntax_errors"] += 1
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            stats["classes"] += 1
            if "Mixin" in node.name:
                mixins.append((str(path), node.lineno, node.name))
            if has_dataclass_decorator(node):
                dataclass_files[str(path)] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stats["functions"] += 1
            size = (node.end_lineno or node.lineno) - node.lineno
            if size > 100:
                big_funcs.append((size, f"{path}:{node.lineno}:{node.name}"))
            ds = docstring_of(node)
            if ds and node.name.startswith("_") and not node.name.startswith("__"):
                if re.search(r"^\s*(Args|Arguments|Parameters|Returns|Raises|Yields)\s*:", ds, re.M):
                    private_args_docs.append((str(path), node.lineno, node.name))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("getattr", "hasattr"):
                if f.id == "getattr":
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        getattr_sites.append((str(path), node.lineno))
                elif node.args and isinstance(node.args[1], ast.Constant):
                    hasattr_sites.append((str(path), node.lineno))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            self_attrs[node.attr] += 1

    for lineno, c in comments_of(text):
        if re.match(r"#\s*TODO\b", c) and not re.match(r"#\s*TODO\(", c):
            bare_todo.append((str(path), lineno, c.strip()[:90]))
        if re.search(r"#\s*(FIXME|XXX|HACK)\b", c):
            bad_tags.append((str(path), lineno, c.strip()[:90]))
        if CJK.search(c) or FANCY.search(c):
            nonascii_comments[str(path)] += 1


for p in sorted(ROOT.rglob("*.py")):
    s = str(p)
    if "/third_party/" in s or "/.git/" in s or s.endswith("_pb2.py"):
        continue
    check_file(p)

# ScheduleBatch direct fields
sb_path = pathlib.Path("python/sglang/srt/managers/schedule_batch.py")
fields = set()
if sb_path.exists():
    try:
        t = ast.parse(sb_path.read_text(encoding="utf-8"))
        for node in ast.walk(t):
            if isinstance(node, ast.ClassDef) and node.name == "ScheduleBatch":
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        fields.add(st.target.id)
    except SyntaxError:
        pass

if fields:
    attr_pat = re.compile(r"\b(self|batch|self\.batch)\.(" + "|".join(sorted(fields)) + r")\b")
    inplace = re.compile(r"\.(extend|append|add_|fill_|zero_|mul_|copy_|scatter_|index_put|resize_|clamp_|copy_)\(|\+=|-=|\|=|&=")
    assign_slice = re.compile(r"\.(extend_seq_lens|seq_lens|input_ids|out_cache_loc|extend_lens)\[")
    for p in sorted(pathlib.Path("python/sglang/srt").rglob("*.py")):
        if p == sb_path:
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if attr_pat.search(line) and (inplace.search(line) or assign_slice.search(line)):
                inplace_sched.append((str(p), i, line.strip()[:110]))

print("== totals ==")
for k in sorted(stats):
    print(f"{k:12} {stats[k]}")
print(f"\n== ScheduleBatch direct fields parsed: {len(fields)} ==")
print(f"\n== files > 2000 LOC: {len(big_files)} ==")
for n, f in sorted(big_files, reverse=True)[:15]:
    print(f"  {n:6} {f}")
print(f"\n== functions > 100 LOC: {len(big_funcs)} ==")
for n, f in sorted(big_funcs, reverse=True)[:15]:
    print(f"  {n:5} {f}")
print(f"\n== @dataclass classes: {sum(dataclass_files.values())} in {len(dataclass_files)} files ==")
for f, n in dataclass_files.most_common(8):
    print(f"  {n:3} {f}")
print(f"\n== defensive getattr(obj,'x'[,d]): {len(getattr_sites)} ==")
for f, l in getattr_sites[:8]:
    print(f"  {f}:{l}")
print(f"== hasattr(obj,'x'): {len(hasattr_sites)} ==")
print(f"\n== bare TODO (no owner): {len(bare_todo)} ==")
for f, l, s in bare_todo[:8]:
    print(f"  {f}:{l} {s}")
print(f"== FIXME/XXX/HACK: {len(bad_tags)} ==")
for f, l, s in bad_tags[:8]:
    print(f"  {f}:{l} {s}")
print(f"\n== non-ASCII/fancy comment files: {len(nonascii_comments)} (sites {sum(nonascii_comments.values())}) ==")
for f, n in nonascii_comments.most_common(8):
    print(f"  {n:4} {f}")
print(f"\n== Args:/Returns: on private helpers: {len(private_args_docs)} ==")
for f, l, n in private_args_docs[:8]:
    print(f"  {f}:{l} {n}")
print(f"\n== Mixin classes: {len(mixins)} ==")
for f, l, n in mixins[:12]:
    print(f"  {f}:{l} {n}")
print(f"\n== ScheduleBatch in-place mutation outside its file: {len(inplace_sched)} ==")
for f, l, s in inplace_sched[:20]:
    print(f"  {f}:{l} {s}")
print("\n== top self.<attr> references ==")
for a, n in self_attrs.most_common(12):
    print(f"  {n:6} self.{a}")

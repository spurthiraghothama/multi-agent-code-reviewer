from reviewer.github.diff import parse_patch, split_git_diff

PATCH = """@@ -10,6 +10,7 @@ def paginate(items, size):
     pages = []
-    for i in range(0, len(items) - size, size):
+    for i in range(0, len(items), size):
         pages.append(items[i:i + size])
+    log(pages)
     return pages
"""


def test_added_and_commentable_lines():
    added, commentable = parse_patch(PATCH)
    assert added == [11, 13]
    assert commentable == [10, 11, 12, 13, 14]


def test_pure_deletion_gets_anchor_line():
    patch = """@@ -5,4 +5,3 @@
 def f(x):
-    if x is None:
-        return 0
     return x + 1
"""
    added, _ = parse_patch(patch)
    assert added == [6]  # the line that now sits where the guard used to be


def test_split_git_diff():
    diff = """diff --git a/pkg/a.py b/pkg/a.py
index 1..2 100644
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,1 @@
+y = 1
"""
    files = split_git_diff(diff)
    assert set(files) == {"pkg/a.py", "pkg/new.py"}
    assert files["pkg/new.py"][0] == "added"
    assert files["pkg/a.py"][1].startswith("@@")
    assert parse_patch(files["pkg/new.py"][1])[0] == [1]

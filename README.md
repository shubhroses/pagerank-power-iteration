# pagerank-power-iteration

PageRank with teleportation, computed by power iteration on a dense transition matrix with NumPy. The implementation is one Python file of about 90 lines, and three small sample graphs come with it.

It was written in December 2022 as a group assignment for a course. The repository was called `5180GroupAssignement4` until October 2026, and the code folder is still named `groupassignment4`. `pagerank.py` and the sample files are unchanged since 2022. This README and the tests were written in October 2026, when `.gitignore` and the NumPy and pytest entries in `pyproject.toml` were also added.

## What the code does

`groupassignment4/pagerank/pagerank.py` defines a class `pagerank`. Its method `pagerank(input_file, alpha=0.15)`:

1. reads a directed graph from a text file and builds its adjacency matrix;
2. turns each row into a probability distribution: a page with out-links spreads its probability evenly over them, and a page with no out-links moves to every page with probability 1/n;
3. applies teleportation, so every entry `p` becomes `(1 - alpha) * p + alpha / n`;
4. starts from the uniform distribution and multiplies it by the transition matrix 14 times (the count is fixed in the code, and a convergence check on the summed change between iterations is present but commented out);
5. prints the vector before each multiplication, then the pages sorted by rank, highest first.

`alpha` is the probability of teleporting to a random page, so the default of 0.15 corresponds to a damping factor of 0.85. The method prints its result and returns `None`.

## Input format

```
<number of pages>
<number of links>
<source page> <target page>
...
```

Pages are numbered from 0. The two numbers on a link line can be separated by spaces or tabs. The link count on the second line is not used by the code, and a repeated link counts once.

The sample graphs are in `groupassignment4/pagerank/`:

| File | Pages | Links | Notes |
| --- | --- | --- | --- |
| `test1.txt` | 5 | 9 | The second line says 10, but 9 links are listed. |
| `test2.txt` | 10 | 15 | Pages 6 and 7 have no out-links, so the 1/n rule of step 2 applies to them. |
| `test3.txt` | 7 | 14 | Five pages link to themselves. This is the graph of a textbook example, see below. |

## Running

`pyproject.toml` declares Python 3.10 or later and NumPy 1.23 or later. With `venv` and `pip`:

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
cd groupassignment4/pagerank
python pagerank.py
```

The script has to be started from that folder because it opens `test3.txt` by a relative path. With Poetry, `poetry install` followed by `poetry run python pagerank.py` in the same folder does the same. `poetry install` takes the package versions from `poetry.lock`.

As committed, the script ranks `test3.txt` with alpha = 0.14. It prints the header line `For file test3.txt`, then 14 vectors, then the ranking. Each vector has seven entries and is wrapped onto two lines. The ranking is shown here rounded to four decimals (the script prints the values at full precision):

```
Page id: 6, Page rank: 0.3059
Page id: 3, Page rank: 0.2456
Page id: 4, Page rank: 0.2132
Page id: 2, Page rank: 0.1127
Page id: 0, Page rank: 0.0524
Page id: 5, Page rank: 0.0351
Page id: 1, Page rank: 0.0351
```

Pages 5 and 1 have the same rank. Ties are printed with the higher page id first.

Fourteen iterations is short of convergence for this graph. With the count raised to 100 the values move by up to 0.0007 (page 6 becomes 0.3066 and page 2 becomes 0.1120), so the numbers above are settled to two decimal places, not four.

To rank another file from Python, in the same folder:

```python
from pagerank import pagerank

pagerank().pagerank("test1.txt", alpha=0.15)
```

## Tests

`tests/test_pagerank.py` was added in October 2026. The assignment had no tests. `pagerank()` returns nothing, so the tests read the ranking from the lines it prints. pytest runs the six test functions as nine cases:

| Test | What it checks |
| --- | --- |
| `test_sample_graph`, once for each sample graph | The ranks sum to 1 within 1e-9 and the pages come in the expected order. |
| `test_sample_ranks_match_saved_output`, for `test1.txt` and `test2.txt` | Each rank is within 0.00005 of the one saved in `out.txt`. |
| `test_textbook_example` | With alpha = 0.14, the ranks of `test3.txt` round to the two-decimal vector printed in the textbook. |
| `test_two_node_cycle` | Two pages link to each other. The transition matrix matches the one worked out by hand, and both ranks are 1/2. |
| `test_chain_with_dead_end` | Page 0 links to page 1, page 1 links to page 2, and page 2 has no out-links. The transition matrix matches the one worked out by hand, the order is 2, 1, 0, and each rank is within 0.00001 of the fixed point, which is proportional to `(1, 1 + b, 1 + b + b^2)` with `b = 1 - alpha`. |
| `test_script_output` | `python pagerank.py`, started in its folder, prints the header line, 14 vectors and the ranking listed above. |

Pages with equal rank are accepted in either order, because the values computed for them can differ in the last digit. On macOS (arm64), the cycle of two pages gives 0.49999999999999944 for page 1 and 0.4999999999999994 for page 0, and that digit decides which page is printed first.

To run the tests, from the repository root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest
pytest
```

With Poetry, `poetry install` followed by `poetry run pytest` does the same.

The pytest settings are in `pyproject.toml`, which also declares pytest 8.0 or later as a development dependency. The settings put `groupassignment4/pagerank` on the import path, and they silence the warning that NumPy gives when `pagerank.py` builds a `numpy.matrix` (see Limitations).

The tests pass on macOS (arm64) with pytest 8.0.0 and with pytest 9.1.1, under the four combinations of Python and NumPy listed in the next section.

## How the results were checked

The following was checked by hand in October 2026. The tests repeat the comparisons of ranks in the first two points.

- Against a published example. `test3.txt` is the seven-page web graph of the worked example in the section "The PageRank computation" of *Introduction to Information Retrieval* by Manning, Raghavan and Schütze (Cambridge University Press, 2008, chapter 21; [online edition](https://nlp.stanford.edu/IR-book/html/htmledition/the-pagerank-computation-1.html)), and 0.14 is the teleportation rate used there. The book prints the transition matrix and the PageRank vector (0.05 0.04 0.11 0.25 0.21 0.04 0.31) to two decimals. The matrix this code builds and the ranks it prints for pages 0 to 6 round to the same numbers.
- Against the saved output. `groupassignment4/out.txt` holds the rankings of `test1.txt` and `test2.txt` printed by commit `326225a`, which ran 100 iterations with alpha = 0.15. Re-running that commit reproduces the file digit for digit. The current 14-iteration code gives the same order, with every rank within 0.00005 of the saved one.
- Across versions. The output for all three graphs was identical under Python 3.10.18 with NumPy 1.23.5 and 2.2.6, and under Python 3.13.7 and 3.14.6 with NumPy 2.5.3, on macOS (arm64). The Poetry route was run with Poetry 2.5.1 on Python 3.10.18, 3.13.7 and 3.14.6: `poetry install` from `poetry.lock`, then the script and the tests. The lock file pins NumPy 2.2.6, the newest release that still supports Python 3.10. That release has no prebuilt wheel for Python 3.14, so there it is compiled from source. On Python 3.10, Poetry warns that it skips a `__pycache__` file contained in the NumPy wheel, and the installation still completes.

## Limitations

- The iteration count is fixed at 14 and nothing checks convergence.
- The transition matrix is a dense n by n `numpy.matrix`, so memory grows with the square of the number of pages. The NumPy documentation no longer recommends the `matrix` class and says it may be removed. It still works in NumPy 2.5.3.
- Input is not validated. A blank line, a line that is not two integers or a page number of n or more raises an exception. A negative page number is silently counted from the end.

## Other files

- `groupassignment4/practice/p.ipynb`: a scratch notebook. It reads `test1.txt` into an adjacency matrix, builds the transition matrix with teleportation for the seven-page graph and holds the helper functions under their earlier camelCase names. Its cells were run out of order and one of them ends in a shape error. Nothing else uses it.
- `.gitattributes`: marks the notebook as documentation for Linguist, the library that computes GitHub's language statistics. Without that line GitHub labels the repository as Jupyter Notebook, because the notebook is larger than the Python code.
- `groupassignment4/out.txt`: the saved 100-iteration output described above.
- `pyproject.toml`: Poetry metadata and the pytest settings. `groupassignment4/__init__.py` is empty and marks the folder as the package that the metadata names.
- `poetry.lock`: the package versions that `poetry install` installs, as resolved by Poetry 2.5.1 in October 2026. They are NumPy 2.2.6, pytest 9.1.1 and the packages that pytest depends on.

## Credits

- The 2022 commits are by Shubhrose Singh and [henryh1570](https://github.com/henryh1570), whose commit `b10d03b` checked the math, added the comments and renamed the variables.
- The class name `pagerank`, the method signature `pagerank(self, input_file)` and the comments directly around them come from the assignment's starter file, which is all that `pagerank.py` contains in the first commit. The `alpha` parameter was added afterwards. `test1.txt` and `test2.txt` were committed with the starter file.
- `test3.txt` reproduces the textbook example cited above.

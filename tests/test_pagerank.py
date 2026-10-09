"""Tests for groupassignment4/pagerank/pagerank.py.

pagerank().pagerank() prints its result and returns None, so these tests read
the ranking from the "Page id: ..., Page rank: ..." lines that it prints.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pagerank import pagerank

PAGERANK_DIR = Path(__file__).resolve().parent.parent / "groupassignment4" / "pagerank"
SAVED_OUTPUT = PAGERANK_DIR.parent / "out.txt"

RANK_LINE = re.compile(r"^Page id: (\d+), Page rank: (\S+)$", re.MULTILINE)

# For each sample graph: the teleportation rate to use and the expected order
# of the pages, highest rank first, with pages of equal rank in one tuple.
#
# The orders of test1 and test2 are those of out.txt, the output saved in 2022.
# Pages 6 and 7 of test2 have no out-links, and pagerank.py fills their rows of
# the transition matrix with 1/n. With those rows left empty the ranks of test2
# would sum to less than 1. The order of test3 is that of the PageRank vector
# printed in the textbook, see test_textbook_example.
SAMPLE_GRAPHS = {
    "test1.txt": (0.15, [0, 2, 3, 1, 4]),
    "test2.txt": (0.15, [8, 9, 5, 3, 0, 1, 2, 7, 4, 6]),
    "test3.txt": (0.14, [6, 3, 4, 2, 0, (5, 1)]),
}


def parse_ranking(text):
    """Return the (page id, rank) pairs printed in text, in the order printed."""
    return [(int(page), float(rank)) for page, rank in RANK_LINE.findall(text)]


@pytest.fixture
def ranking_of(capsys):
    """Return a function that ranks a graph file and returns the printed ranking."""

    def run(graph_file, **options):
        pagerank().pagerank(graph_file, **options)
        return parse_ranking(capsys.readouterr().out)

    return run


def write_graph(folder, text):
    """Write a graph in the input format to a file in folder and return its path."""
    path = folder / "graph.txt"
    path.write_text(text)
    return path


def assert_ranking(ranking, expected_order):
    """Check that the ranks sum to 1 and that the pages come in expected_order.

    expected_order lists the page ids from highest rank to lowest. Pages of
    equal rank go in one tuple and may be printed in either order: their
    computed values can differ in the last digit, and that digit then decides
    which of them is printed first.
    """
    assert abs(sum(rank for _, rank in ranking) - 1) < 1e-9

    start = 0
    for entry in expected_order:
        tied = entry if isinstance(entry, tuple) else (entry,)
        printed = ranking[start : start + len(tied)]
        assert {page for page, _ in printed} == set(tied)
        ranks = [rank for _, rank in printed]
        assert max(ranks) - min(ranks) < 1e-12
        start += len(tied)
    assert start == len(ranking)


@pytest.mark.parametrize("name", SAMPLE_GRAPHS)
def test_sample_graph(name, ranking_of):
    """The ranks of a sample graph sum to 1 and come in the expected order."""
    alpha, expected_order = SAMPLE_GRAPHS[name]
    assert_ranking(ranking_of(PAGERANK_DIR / name, alpha=alpha), expected_order)


def saved_ranks(name):
    """Return {page id: rank} for one sample graph from out.txt.

    out.txt was printed in 2022 by an earlier version of pagerank.py, which ran
    100 iterations on test1.txt and test2.txt with the default alpha.
    """
    after_header = SAVED_OUTPUT.read_text().split(f"For file {name}\n")[1]
    section = after_header.split("For file ")[0]
    return dict(parse_ranking(section))


@pytest.mark.parametrize("name", ["test1.txt", "test2.txt"])
def test_sample_ranks_match_saved_output(name, ranking_of):
    """The 14 iterations run today end within 0.00005 of the saved ranks."""
    ranks = dict(ranking_of(PAGERANK_DIR / name))
    assert ranks == pytest.approx(saved_ranks(name), abs=5e-5)


def test_textbook_example(ranking_of):
    """test3.txt is the graph of a worked example in a textbook.

    The example is in the section "The PageRank computation" of Introduction
    to Information Retrieval by Manning, Raghavan and Schütze (2008). For a
    teleportation rate of 0.14 the book prints the PageRank vector below, to
    two decimals.
    """
    ranks = dict(ranking_of(PAGERANK_DIR / "test3.txt", alpha=0.14))
    rounded = [round(ranks[page], 2) for page in range(7)]
    assert rounded == [0.05, 0.04, 0.11, 0.25, 0.21, 0.04, 0.31]


def test_two_node_cycle(tmp_path, ranking_of):
    """0 -> 1 and 1 -> 0.

    With alpha = 0.15 a page moves to the other page with probability
    0.85 + 0.15 / 2 and stays where it is with probability 0.15 / 2. The two
    pages are interchangeable, so each has rank 1/2 after every iteration.
    """
    graph = write_graph(tmp_path, "2\n2\n0 1\n1 0\n")

    matrix = pagerank().get_transition_mat_with_tp(graph, 0.15)
    np.testing.assert_allclose(
        matrix, [[0.075, 0.925], [0.925, 0.075]], rtol=0, atol=1e-12
    )

    ranking = ranking_of(graph, alpha=0.15)
    assert_ranking(ranking, [(0, 1)])
    assert dict(ranking) == pytest.approx({0: 0.5, 1: 0.5}, abs=1e-12)


def test_chain_with_dead_end(tmp_path, ranking_of):
    """0 -> 1 -> 2, and page 2 has no out-links.

    Page 2 is a dead end, so it moves to each of the three pages with
    probability 1/3. Write b = 1 - alpha. The fixed point x = xP satisfies

        x0 = alpha / 3 + b * x2 / 3    (teleporting, or leaving the dead end)
        x1 = x0 + b * x0               (the same, plus the link from page 0)
        x2 = x0 + b * x1               (the same, plus the link from page 1)

    so x is proportional to (1, 1 + b, 1 + b + b^2). For alpha = 0.15 that is
    about (0.1844, 0.3412, 0.4744). On this graph the 14 iterations of
    pagerank() end within 0.00001 of the fixed point.
    """
    alpha = 0.15
    graph = write_graph(tmp_path, "3\n2\n0 1\n1 2\n")

    matrix = pagerank().get_transition_mat_with_tp(graph, alpha)
    np.testing.assert_allclose(
        matrix,
        [[0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [1 / 3, 1 / 3, 1 / 3]],
        rtol=0,
        atol=1e-12,
    )

    ranking = ranking_of(graph, alpha=alpha)
    assert_ranking(ranking, [2, 1, 0])

    b = 1 - alpha
    weights = [1, 1 + b, 1 + b + b**2]
    fixed_point = {page: weight / sum(weights) for page, weight in enumerate(weights)}
    assert dict(ranking) == pytest.approx(fixed_point, abs=1e-5)


def test_script_output():
    """Run pagerank.py from its own folder, as the README describes.

    The script ranks test3.txt with alpha = 0.14. It prints a header line, then
    the vector before each of the 14 multiplications, then the ranking.
    """
    result = subprocess.run(
        [sys.executable, "pagerank.py"],
        cwd=PAGERANK_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # test3.txt has seven pages, so the ranking is the last seven lines.
    lines = result.stdout.splitlines()
    header, vector_lines, rank_lines = lines[0], lines[1:-7], lines[-7:]
    assert header == "For file test3.txt"
    # Each vector starts a line with "[". One that is too long for a single
    # line continues on an indented line.
    assert sum(line.startswith("[") for line in vector_lines) == 14

    ranking = parse_ranking("\n".join(rank_lines))
    _, expected_order = SAMPLE_GRAPHS["test3.txt"]
    assert_ranking(ranking, expected_order)
    # The ranks listed in the README, which rounds them to four decimals.
    assert {page: round(rank, 4) for page, rank in ranking} == {
        6: 0.3059,
        3: 0.2456,
        4: 0.2132,
        2: 0.1127,
        0: 0.0524,
        5: 0.0351,
        1: 0.0351,
    }

# =============================================================================
#  00.01 -- Hello, tests
# =============================================================================
#
#  Every exercise in this course is a single self-contained file that both
#  states the problem and checks your answer. The code you edit is at the top,
#  the tests are at the bottom, and pytest runs the tests to find out whether
#  you were right. There is nothing else to read: the comment you are reading
#  now is the teaching material.
#
#  The tests are ordinary functions whose names start with `test_`. pytest
#  finds them, runs them, and treats a failing `assert` as a failing test.
#  When one fails it prints the values on both sides of the comparison, which
#  is most of what you need to fix it (00.02 is about reading that output).
#
#  TASK
#    Make `greet` return "Hello, " followed by the name and an exclamation
#    mark, so that `greet("world")` is "Hello, world!".
#
#  RUN IT
#    ./npt test 00_01
#
# =============================================================================


def greet(name: str) -> str:
    """Return a greeting for `name`."""
    # TODO: build the greeting.
    #
    # Use an f-string. It is the fastest and most readable way to put values
    # into text, and it is what the rest of this course uses (Effective Python,
    # 3rd ed., Item 11: "Prefer Interpolated F-Strings over C-Style Format
    # Strings and str.format").
    return "Hello!"


def test_greet_builds_a_greeting():
    assert greet("world") == "Hello, world!"
    assert greet("modern NumPy") == "Hello, modern NumPy!"


def test_greet_does_not_alter_the_name():
    # A plausible shortcut is `.title()` or `.strip()` on the name. The function
    # is not asked to tidy anything up, and a test that only used tidy inputs
    # would never notice if it did.
    assert greet("  spaced out  ") == "Hello,   spaced out  !"
    assert greet("") == "Hello, !"

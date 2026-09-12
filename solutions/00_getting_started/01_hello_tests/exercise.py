# Solution -- 00.01 Hello, tests


def greet(name: str) -> str:
    """Return a greeting for `name`."""
    return f"Hello, {name}!"


def test_greet_builds_a_greeting():
    assert greet("world") == "Hello, world!"
    assert greet("modern NumPy") == "Hello, modern NumPy!"


def test_greet_does_not_alter_the_name():
    # A plausible shortcut is `.title()` or `.strip()` on the name. The function
    # is not asked to tidy anything up, and a test that only used tidy inputs
    # would never notice if it did.
    assert greet("  spaced out  ") == "Hello,   spaced out  !"
    assert greet("") == "Hello, !"

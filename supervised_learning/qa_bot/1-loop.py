#!/usr/bin/env python3
"""Task 1"""


def main():
    """
    Create a script that takes in input from the user with the prompt and prints A: as a response
    """
    while True:
    x = input("Q: ").lower()
    if x in {"exit", "quit", "goodbye", "bye"}:
        print("A: Goodbye")
        break
    print("A:")


if __name__ == "__main__":
    main()

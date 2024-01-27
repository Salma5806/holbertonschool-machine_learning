#!/usr/bin/env python3
"""Task 1"""


def main():
    """
    Create a script that takes in input from the user with the prompt Q: 
    and prints A: as a response
    """
    while True:
        # Prompt the user for input and convert it to lowercase
        user_input = input("Q: ").lower()

        # If the user input is an exit command, print
        # goodbye message and exit the loop
        if user_input in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            # If the user input is not an exit command,
            # print a placeholder response
            print("A:")


if __name__ == "__main__":
    main()

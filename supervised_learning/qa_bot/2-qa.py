#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

question_answer = __import__('0-qa').question_answer

def answer_loop(reference):
    """Answers questions based on reference text until exit command or interrupt"""
    EXIT_COMMANDS = {'exit', 'quit', 'goodbye', 'bye'}
    
    try:
        while True:
            question = input("Q: ").strip()
            
            if question.lower() in EXIT_COMMANDS:
                print("A: Goodbye")
                return
                
            answer = question_answer(question, reference)
            print(f"A: {answer}" if answer else "A: Sorry, I do not understand your question.")
            
    except (KeyboardInterrupt, EOFError):
        print("\nA: Goodbye")

#!/bin/bash
echo "Running tests..."
python3 -m unittest test/test_simulate.py
echo "All tests passed."

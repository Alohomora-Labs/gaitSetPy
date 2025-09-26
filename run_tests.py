#!/usr/bin/env python3
"""
Test runner script for GaitSetPy comprehensive testing framework.

This script provides a convenient way to run different categories of tests
and generate test reports for the GaitSetPy package.

Usage:
    python run_tests.py [options]

Options:
    --unit          Run only unit tests
    --integration   Run only integration tests
    --all           Run all tests (default)
    --coverage      Generate coverage report
    --verbose       Verbose output
    --fast          Skip slow tests

Maintainer: @aharshit123456
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ FAILED")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="GaitSetPy Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--file", help="Run specific test file")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        base_cmd.append("-v")
    else:
        base_cmd.append("-q")
    
    # Add coverage if requested
    if args.coverage:
        base_cmd.extend(["--cov=gaitsetpy", "--cov-report=html", "--cov-report=term"])
    
    # Skip slow tests if requested
    if args.fast:
        base_cmd.extend(["-m", "not slow"])
    
    # Determine test paths
    if args.file:
        test_paths = [args.file]
    elif args.unit:
        test_paths = ["tests/unit/"]
    elif args.integration:
        test_paths = ["tests/integration/"]
    else:
        # Run all tests except those with naming conflicts
        test_paths = [
            "tests/unit/test_base_classes.py",
            "tests/unit/test_managers.py", 
            "tests/unit/test_classification.py",
            "tests/test_daphnet.py",
            "tests/test_harup.py",
            "tests/test_physionet.py",
            "tests/integration/"
        ]
    
    # Add test paths to command
    base_cmd.extend(test_paths)
    
    # Run the tests
    success = run_command(base_cmd, "GaitSetPy Test Suite")
    
    if success:
        print(f"\n🎉 All tests passed successfully!")
        if args.coverage:
            print(f"📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

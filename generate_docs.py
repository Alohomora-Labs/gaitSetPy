#!/usr/bin/env python3
"""
Documentation generation script for gaitsetpy package.

This script uses pdoc to generate HTML documentation for the entire gaitsetpy package.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    """Generate HTML documentation using pdoc."""
    
    # Get the current directory
    current_dir = Path.cwd()
    
    # Check if gaitsetpy package exists
    gaitsetpy_path = current_dir / "gaitsetpy"
    if not gaitsetpy_path.exists():
        print("Error: gaitsetpy package not found in current directory")
        sys.exit(1)
    
    print("Generating documentation for gaitsetpy package...")
    
    try:
        # Generate documentation using pdoc
        # This will create HTML files for all modules and submodules
        result = subprocess.run([
            sys.executable, "-m", "pdoc",
            "-o", ".",
            "gaitsetpy"
        ], check=True, capture_output=True, text=True)
        
        print("Documentation generated successfully!")
        print(f"Output: {result.stdout}")
        
        # Create index.html redirect file
        index_content = '''<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=./gaitsetpy.html" />
  </head>
</html>
'''
        
        with open("index.html", "w") as f:
            f.write(index_content)
        
        print("Created index.html redirect file")
        
        # List generated files
        html_files = list(current_dir.rglob("*.html"))
        print(f"\nGenerated {len(html_files)} HTML files:")
        for html_file in sorted(html_files):
            print(f"  - {html_file.relative_to(current_dir)}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating documentation: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
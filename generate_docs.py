#!/usr/bin/env python3
"""
Documentation generation script for gaitsetpy package.

This script uses pdoc to generate HTML documentation for the entire gaitsetpy package.
It dynamically discovers modules from the package structure and __init__.py files.
"""

import os
import subprocess
import sys
import shutil
import pkgutil
import importlib
from pathlib import Path


def discover_modules(package_name):
    """
    Dynamically discover all modules in a package.
    
    Args:
        package_name: Name of the package to discover modules for
        
    Returns:
        List of module names that can be imported
    """
    modules = []
    
    try:
        # Import the main package
        package = importlib.import_module(package_name)
        modules.append(package_name)
        
        # Walk through all submodules using pkgutil
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, 
            package.__name__ + "."
        ):
            try:
                # Try to import the module to verify it's valid
                importlib.import_module(modname)
                modules.append(modname)
                print(f"  ✅ Discovered: {modname}")
            except ImportError as e:
                print(f"  ❌ Skipped: {modname} ({e})")
            except Exception as e:
                print(f"  ⚠️  Warning: {modname} ({e})")
                
    except ImportError as e:
        print(f"Error: Could not import package '{package_name}': {e}")
        sys.exit(1)
    
    return modules


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
        # Dynamically discover all modules in the gaitsetpy package
        print("Discovering all Python modules in gaitsetpy...")
        modules_to_document = discover_modules("gaitsetpy")
        
        print(f"\nFound {len(modules_to_document)} modules to document")
        
        # Generate documentation using pdoc
        cmd = [
            sys.executable, "-m", "pdoc",
            "-o", ".",
            "--docformat", "restructuredtext",
            "--include-undocumented"
        ] + modules_to_document
        
        print(f"\nGenerating documentation for {len(modules_to_document)} modules...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Documentation generated successfully!")
        if result.stdout:
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
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
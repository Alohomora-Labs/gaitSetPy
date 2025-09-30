---
name: Documentation Improvement
about: Improve or add documentation
title: '[DOCS] '
labels: 'documentation'
assignees: ''
---

## Documentation Type

- [ ] API Reference
- [ ] Tutorial
- [ ] Example
- [ ] README Update
- [ ] Docstring Addition/Improvement
- [ ] Guide/How-to
- [ ] Other

## Description

[Describe what documentation needs to be added or improved]

## Current State

**What exists:**
[Describe current documentation state]

**What's missing/needs improvement:**
[Describe gaps or issues]

## Proposed Changes

### Files to Create/Modify
- [ ] `path/to/doc1.md` - [description]
- [ ] `path/to/doc2.py` - [docstring improvements]
- [ ] `examples/example.py` - [example script]

### Content Outline
1. [Section 1]
   - [Subsection 1.1]
   - [Subsection 1.2]
2. [Section 2]
   - [Subsection 2.1]

## Tasks

### For API Documentation:
- [ ] Add/improve docstrings following Google style
- [ ] Add type hints to all parameters
- [ ] Include usage examples in docstrings
- [ ] Document all parameters, returns, and exceptions
- [ ] Verify documentation builds correctly

### For Tutorials:
- [ ] Write tutorial content
- [ ] Add code examples that run correctly
- [ ] Add visualizations/diagrams if helpful
- [ ] Test all examples
- [ ] Add to documentation index

### For Examples:
- [ ] Create example script
- [ ] Add comprehensive comments
- [ ] Test example runs without errors
- [ ] Add to examples directory
- [ ] Reference in README/documentation

### For README:
- [ ] Update installation instructions
- [ ] Update usage examples
- [ ] Update feature list
- [ ] Update badges/status
- [ ] Check all links work

## Documentation Standards

### Docstring Format (Google Style)
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief one-line description.
    
    Detailed description of what the function does, including
    any important notes or considerations.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: Description of when this is raised
        TypeError: Description of when this is raised
    
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        Expected output
    """
    pass
```

### Example Script Structure
```python
"""
Example: [Title]

This example demonstrates:
1. [Key point 1]
2. [Key point 2]
3. [Key point 3]

Requirements:
- [Requirement 1]
- [Requirement 2]

Author: [Your name]
"""

import gaitsetpy as gsp

# Step 1: [Description]
# [Code with comments]

# Step 2: [Description]  
# [Code with comments]

# Output/Results
print("[Description of output]")
```

### Markdown Documentation
- Use clear headings (H1, H2, H3)
- Include code examples with syntax highlighting
- Add tables for structured information
- Include diagrams/images where helpful
- Link to related documentation
- Add table of contents for long documents

## Target Audience

- [ ] Beginners (new to gait analysis)
- [ ] Intermediate (familiar with gait analysis)
- [ ] Advanced (extending the package)
- [ ] API users
- [ ] Contributors

## Related Documentation

[Link to related documentation pages]

## Examples to Include

```python
# Example code that should be documented
import gaitsetpy as gsp

# Example usage
...
```

## Checklist

- [ ] Content is accurate and up-to-date
- [ ] Code examples run without errors
- [ ] Links are working
- [ ] Images/diagrams are clear and helpful
- [ ] Grammar and spelling checked
- [ ] Follows documentation style guide
- [ ] Documentation builds successfully
- [ ] Reviewed by another person (if possible)

## Building Documentation

```bash
# Generate API documentation
python generate_docs.py

# View documentation locally
# Open gaitsetpy.html in browser
```

## Definition of Done

- [ ] Documentation is complete and accurate
- [ ] All code examples tested and working
- [ ] Documentation builds without errors
- [ ] Links are working
- [ ] Reviewed and approved
- [ ] Merged to main branch

## Additional Notes

[Add any additional context or requirements]

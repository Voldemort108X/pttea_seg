"""models package initializer

This file turns the models/ directory into a Python package so that
imports like `from models.segmentation import ...` work when the project
root is on sys.path.
"""

__all__ = ["segmentation", "modelio", "energy"]

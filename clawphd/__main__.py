"""
Entry point for running clawphd as a module: python -m clawphd
"""

from clawphd.cli.commands import app

if __name__ == "__main__":
    app()

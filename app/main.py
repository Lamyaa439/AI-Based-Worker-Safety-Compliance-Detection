"""
Main entry point for the application.
Run: python -m app.main  (from the project's root directory)
"""

from app.ui.interface import build_interface
 
if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
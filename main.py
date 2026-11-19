"""Start the application:  python main.py"""
import sys

if sys.version_info < (3, 10):
    sys.exit("Potrzebny jest Python 3.10 lub nowszy.")

try:
    from app.gui.main_window import run
except ImportError as exc:
    sys.exit(f"Brakuje biblioteki {exc.name}. Zainstaluj zależności:\n    pip install -r requirements.txt")

if __name__ == "__main__":
    sys.exit(run())

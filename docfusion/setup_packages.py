import subprocess
import sys

# List of packages you need
packages = [
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "pytesseract", "pillow", "opencv-python-headless",
    "jupyter", "streamlit",
    "textblob", "fuzzywuzzy", "python-Levenshtein"
]

missing_packages = []

# Check each package
for pkg in packages:
    try:
        __import__(pkg if pkg != "pillow" else "PIL")  # pillow is imported as PIL
        print(f"{pkg} ✅ already installed")
    except ImportError:
        print(f"{pkg} ❌ missing")
        missing_packages.append(pkg)

# Install missing packages
if missing_packages:
    print("\nInstalling missing packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
    print("\n✅ All missing packages installed!")
else:
    print("\nAll packages are already installed! 🎉")
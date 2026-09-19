# Creating a venv and installing dependencies in one file because I'm lazy to type it out every time
#!/usr/bin/env bash

set -e

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"

    echo "Activating virtual environment..."
    source "$VENV_DIR/Scripts/activate"

    if [ -f "requirements.txt" ]; then
        echo "Installing libraries..."
        pip install -r requirements.txt
    else
        echo "No requirements.txt found, skipping install."
    fi
fi

echo "Done!"

# Run it like this:
# source venv.sh
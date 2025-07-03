#!/bin/bash

### MGFunc Installer
### Updated: MAR. 2025
### Bioinformatics Group, College of Computer Science and Technology, Qingdao University
### Code by: Yu Zhang, Xiaoquan Su

echo "** Installation Start **"

# Detect shell config file
if [[ $SHELL == */zsh ]]; then
    PATH_File="${HOME}/.zshrc"
    [ -f "$PATH_File" ] || PATH_File="${HOME}/.zsh_profile"
else
    PATH_File="${HOME}/.bashrc"
    [ -f "$PATH_File" ] || PATH_File="${HOME}/.bash_profile"
fi
touch "$PATH_File"

# Check platform
if [[ "$(uname)" != "Linux" ]]; then
    echo "This installer supports only Linux."
    return 1 2>/dev/null || exit 1
fi

MGFUNC_PATH=$(pwd)
Add_Part="####DisabledbyMGFunc####"

# Build source code
BUILD_MODE=${1:-cuda}
echo -e "\n** MGFunc Source Build **"

if [ -f "Makefile" ]; then
    echo "** Cleaning old builds **"
    make clean

    if [ "$BUILD_MODE" == "hip" ]; then
        echo "** Building GCC + HIP version **"
        make hip
    else
        echo "** Building GCC + CUDA version **"
        make
    fi

    echo -e "\n** Build Complete **"
else
    echo "** Binary package detected, skipping compilation **"
fi

# Set environment variables
# Disable old MGFunc lines if they exist
if grep -q "^export MGFunc=" "$PATH_File"; then
    sed -i "s|^export MGFunc=.*|$Add_Part &|g" "$PATH_File"
fi

# Add new MGFunc path
if ! grep -qxF "export MGFunc=$MGFUNC_PATH" "$PATH_File"; then
    echo "export MGFunc=$MGFUNC_PATH" >> "$PATH_File"
fi

# Add MGFunc/bin to PATH if not present
if ! grep -qxF 'export PATH=$PATH:$MGFunc/bin' "$PATH_File"; then
    echo 'export PATH=$PATH:$MGFunc/bin' >> "$PATH_File"
fi

# Apply environment variables if sourced
if [[ "$0" != "$BASH_SOURCE" ]]; then
    source "$PATH_File"
    echo -e "\n** Environment Variables Applied to Current Shell **"
else
    echo -e "\n** Environment variables have been written to $PATH_File **"
    echo "** Please run 'source $PATH_File' to apply them to your current shell. **"
fi

# End
echo -e "\n** MGFunc Installation Complete **"
echo "** Example dataset with demo script is available in 'example/' directory **"

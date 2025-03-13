#!/bin/bash

cd datasets
git clone https://github.com/lupantech/ScienceQA

if [ $? -ne 0 ]; then
    echo "Failed to clone the repository. Please check your network or the repository address."
    exit 1
fi

cd ScienceQA
bash tools/download.sh

if [ $? -ne 0 ]; then
    echo "Failed to execute the download script."
    exit 1
fi

cd ..
cd src
cd Image_to_Text
python CLIP_Interrogator_ScienceQA.py

if [ $? -ne 0 ]; then
    echo "Failed to execute the Python script."
    exit 1
fi

echo "All steps executed successfully."
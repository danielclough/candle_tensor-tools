#!/bin/bash

# fail on error
set -e

# Confirm inital setup
read -p "Do you already have ssh and a hf.co repo setup (y/n)? " CONFIRM
if [ "${CONFIRM}" != "y" ]; then
    echo "Setup your SSH and https://hf.co/new repo first!"
else
    # Create New Dir
    read -p "What is the HuggingFace repo username/model? " USERNAME_MODEL
    read -p "Where do you want to create it? " PARENT_DIR

    HF_ID=`echo ${USERNAME_MODEL} | cut -d '/' -f 1`
    MODEL_NAME=`echo ${USERNAME_MODEL} | cut -d '/' -f 2`

    PROJECT_DIR=${PARENT_DIR}/${MODEL_NAME}

    mkdir ${PROJECT_DIR}
    cd ${PROJECT_DIR}

    # Insure dependencies are installed
    if [ -z `which git-lfs` ]; then
        echo "Installing git and git-lfs (requires sudo)"
        sudo apt-get update
        sudo apt-get -y upgrade
        sudo apt-get install git git-lfs
    fi

    # init
    git init

    # Prepare LFS
    git lfs install
    huggingface-cli lfs-enable-largefiles .

    # get remote from hf.co
    git remote add origin git@hf.co:${HF_ID}/${MODEL_NAME}.git
    git pull origin main

    # config lfs for .gguf files
    echo "*.gguf filter=lfs diff=lfs merge=lfs -text" >> .gitattributes

    # push changes
    git add -A
    git commit -m init
    git branch -M main
    git push -u origin main
fi
#!/bin/bash

# this is a script that gathers data for SER

# prepare dirs

cd ~
mkdir -p "SER_project/data/emodb/"

# EMODB

cd "SER_project/data/emodb"

# download dataset
wget -nv "http://emodb.bilderbar.info/download/./download.zip"

# extract archive and filter wav files
unzip -q "download.zip" "wav/*"
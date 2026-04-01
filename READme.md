



# Autonomous Car Project

## Overview
it is the software for an autonomous car 

## Getting Started
1) fork the repo and download FSDS.exe into foreground foreground/engine_binaries and in engine_binaries move seeting.json 


2) your branch must be named (your name)_(the function your solving)

3) make sure you are in your branch 

4) Build the image (recreates the same environment)
docker build -t rust-python .

5) Run the container with the repo mounted
Windows PowerShell
docker run --rm -p 8080:80 -p 8081:81 -p 8082:82 -p 8083:83 -p 8084:84 -it -v "${PWD}:/work" -w /work rust-python bash

macOS/Linux
docker run --rm -p 8080:80 -p 8081:81 -p 8082:82 -p 8083:83 -p 8084:84 -it -v "$(pwd):/work" -w /work rust-python bash


6) run 
python com_window_code/main.py





7) Verify tools inside the container

which python
python --version
pip --version
cargo --version


## Contributing
Please open a Pull Request for all changes.

for ml devs 
once ur done with development make sure u run 
pip freeze 
and copy tht output to requirments.txt 

steps to contribute 

1) push to your branch and open a pull request 
2) this request will be reviewed and approved for merging 

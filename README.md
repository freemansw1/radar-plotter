Readme for plotting radar data

Originally created by Sean Freeman, sean.freeman@colostate.edu

Parameters are up top, they should be fairly straightforward.


## Install Instructions

First, you must install the appropriate libraries. Create a new conda
environment using the included environment file: 
`conda env create -f radar-plotter_env.yml`. 
This will download and install the appropriate libraries.


## Environment Setup Instructions

Once the environment is installed, activate it 
`conda activate radar-plotter`. 
You will have to activate the environment every time that you want to 
run this script. 


## Running Instructions

This script has been parallelized with Jug. Running jug scripts is a 
bit more complicated, but it does allow you to run things in parallel 
across any node that shares a filesystem.

To check current status, run: 
`jug status plot_radar_data.py`. 
That will give you the current status. To start running, 
`jug execute plot_radar_data.py > out.txt 2>&1 &`. 
Run that command for as many processors that you want to run the 
script on. 

Jug will create a `plot_radar_data.jugdata` folder. When you are done 
plotting, delete that folder. You won't be able to run another of 
these scripts until you delete the folder.
 

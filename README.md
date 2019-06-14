# SUDOKU 
Converts image of SUDOKU into digital form and finds a solution


 - Using Python 3.6.8 and OpenCV 4.1.0
 
 ## How to install:
  
   - Install  [Anaconda](https://www.anaconda.com/distribution/)
   
   - Clone this repository ```git clone https://github.com/horczech/sudoku.git```
   
   - Install the virtual environment from ```enviroment.yml``` file located in the root directory of the 
   repository using
    
     ```conda env create -f environment.yml``` 
   
   - Activate the virtual enviroment 
   
     ```conda activate sudoku_env```
  
 ## How to use:
  
   - There are two scripts that can be used from command line. 
   
     1  ```sudoku_solver.py``` that accepts SUDOKU image and returns image with solved SUDOKU and text version of 
   solution. 
   
    ```python3 sudoku_solver.py sudoku_imgs/web_cam/webcam_clean_1.jpg``` 
   
   
   There is also optional parameter ```--config``` where you can specify parameters of the algorithm. The example config files can be found 
   in ```/configs/``` directory. 
   
    ```python3 sudoku_solver.py sudoku_imgs/web_cam/webcam_clean_1.jpg --config configs/config_07```
     
   2 ```camera_sudoku_solver.py``` This script has no input arguments it just finds and solves soduku from the 
     camera stream. By pressing "F" key on the keyboard it will freeze the surrent frame from camera and by pressing 
     "R" key it will return to the camera stream. By pressing "P" key it will wait till it finds valid solution of the 
     SUDOKU and prints image with the solution and text version of solution. Run it using 
     
     ```python3 camera_sudoku_solver.py ```



 ## Sources:
  - http://www.aishack.in/tutorials/sudoku-grabber-opencv-plot/
  - https://hackernoon.com/sudoku-solver-w-golang-opencv-3-2-3972ed3baae2
  - https://ieeexplore.ieee.org/document/7007986?tp=&arnumber=7007986

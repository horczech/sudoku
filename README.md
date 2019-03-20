# SUDOKU 
Converts image of SUDOKU into digital form


 - Using Python 3.7.2 and OpenCV 3.4.2
 
 ## How to use:
   - Use the ```enviroment.yml``` to create virtual environment using the conda's command 
   ```conda env create -f environment.yml```  
   - Run from the commmand line with the image as the only mandatory argument
    ```python3 main.py sudoku_imgs/unannotated_imgs/2.jpg```
   - Create own config file and pass it's path via optional argument (the name of the config file MUST be ```config.py```
   ```python3 main.py sudoku_imgs/unannotated_imgs/2.jpg --config configs/config_1/config.py```
   
   - Run the pytest in the ```tests/``` directory. Each step will visualize individual steps during the image processing
   
   - Parameters of the algorithm can be tuned in the config file located in ```configs/config_1/config.py```
    
 
 ## ToDo:
  - [ ] Tune the parameters and make the algorithm more robust
  - [ ] Find out why the OCR has problem with some numbers (e.g. 1)
  - [ ] Make the algorithm faster (currently it takes 9sec on my PC)
  - [ ] Collect more test images with annotations and make real tests
  - [ ] Find better way how to load config file


 ## Ideas:
  - The OCR model can be trained on font that will be used in praxis 


 ## Sources:
  - http://www.aishack.in/tutorials/sudoku-grabber-opencv-plot/
  - https://hackernoon.com/sudoku-solver-w-golang-opencv-3-2-3972ed3baae2
  - https://ieeexplore.ieee.org/document/7007986?tp=&arnumber=7007986

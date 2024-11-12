Shape Estimation - Main Folder

Description of each Folder:

CAD Files: Contains all relevant Parts as step.file
COMSOL: Contains all Simulation relevant files
Data Collection: All collected data with the test Setup stored by date
Docs: All Documents including Thesis
Figures: Figures used in the Thesis
Shape_Estimation_Test_Setup_Angle_Calculation: Arduino Script for Shape Estimation and Error Anaylsis
Shape_Estimation_Test_Setup_Micro: Arduino Scipt for collecting Look-Up Values

Important Files:
analysis_TestSetupV4.py -> Analyzing Data
animationPlot.py -> Used for Liveplot
serial_read.py -> Read Data send from Shape_Estimation_Test_Setup_Micro
ShapeEstimation_TestSetupV4.py -> Read and send data of Shape_Estimation_Test_Setup_Angle_Calculation

Operation:

How to use ShapeEstimation_TestSetupV4.py:
1. Load Shape_Estimation_Test_Setup_Angle_Calculation to Arduino
2. Run Code
3. Enter all Information (must match KEYS or DISTANCES)
4. Enter Angle

How to use anaylsis_TestSetupV4.py:

1. Data are stored in file such as R1_female_0.2A_200_11.5mm
R1: First Measuremnet cycle
female: Magnetized Ball Joint part
0.2A: Current through Coil
200: Number of coil windings
11.5mm: Height Element used for the sensor, which Maps a specific sensing radius.

2. Make sure all files are in this Folder section. If not, copy from "Data Collection"

How to use serial_read.py:
1. Load Shape_Estimation_Test_Setup_Micro to Arduino
2. Modify last 3 Lines by adjusting Name which the files should be stored
3. Run Code


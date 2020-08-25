----File structure----
In the main folder, the code is written in final.py
All the configuration files are in the subfolder configs
/configs/source- configuration files for source images
	   /target- configuration files for target images
/samples_as3/input/source- source images
	             /target- target images
/output- final output images

----Functions----
-> main()- has code for input interactions, calling necessary functions and writing the final output image.
-> getConfig()- has code for extracting necessary info on images from their configuration files.
-> calculateInitP()- has code for calculating initial parameter matrix.
-> estimateParameters()- has code for calculating final parameter matrix.

----Note----
-> Console asks the user to enter the path to source config file and path to target config file.
-> In the end, it asks the user to enter the path to the final output image.

----Packages Used----
shapely - to check whether point is within the polygon or not.
numpy - for basic matrix manipulations.
imageio - for inputting and outputting images.
tqdm - to check the progress of code.

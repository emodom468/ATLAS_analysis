/* DRAW AND MEASURE BRAIN SURFACE POINTS
 * input folder: folder with only max projection of ch2 files
 * output folder: save points to a directory with file name identifier
*/

inputFolder= getDirectory("Choose Max Proj Directory");
//inputFolder = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/All_Ch2_Max'
//outputFolder = getDirectory("Choose the Parent Directory to Save");
outputFolder = "/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/surface_points/"
print(inputFolder);
images= getFileList(inputFolder);

for (i = 0; i < images.length; i++) {
    setBatchMode(false); // Disable batch mode
    inputPath = inputFolder + images[i];
    print("opening: " + inputPath);
    open(inputPath);

    // Open Brightness/Contrast window, wait for user input
    run("Brightness/Contrast...");

    // Open ROI Manager
    run("ROI Manager...");
    roiManager("reset");

    // Select the oval tool
    setTool("multipoint"); 
    waitForUser("Draw ROIs, then click OK to measure.");
	// Measure
	roiManager("Add"); 
	roiManager("Select", 0);
    //roiManager("Deselect");
    roiManager("Measure");
    print("Rois Measured");
    
    //Save 
    imagesName=File.getNameWithoutExtension(inputPath);
    savePath = outputFolder + imagesName + "_results.csv";
    saveAs("Results", savePath);
    print("Saved: ",savePath);
    
    run("Clear Results"); 
    //run("Save As...", "measurements=[" + measurementFolder + images[i] + "_measurements.csv]");
	run("Close All");
}

print("done");

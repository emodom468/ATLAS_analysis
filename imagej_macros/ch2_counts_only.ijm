inputFolder = getDirectory("Choose a Directory");
print("Input Folder: " + inputFolder);
images = getFileList(inputFolder);

outputFolder = "/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results";
//getDirectory("Choose the Parent Directory to Save Results");
print("Output Folder: " + outputFolder);

roiFolder = outputFolder + "/ROI_zip/";
measurementFolder = outputFolder + "/ROI_results/";

if (!File.isDirectory(roiFolder)) {
    File.makeDirectory(roiFolder);
}


if (!File.isDirectory(measurementFolder)) {
    File.makeDirectory(measurementFolder);
}

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
    setTool("oval");

    // Wait for user to draw and adjust ROIs, then measure
    waitForUser("Draw ROIs, then click OK to measure.");
    //roiManager("Add");
	//print("roi manager added");
    // Save ROIs
    roiSavePath = roiFolder + images[i] + "_RoiSet.zip";
    //roiManager("Select", 0);
    roiManager("Deselect");
    print("roi manager selected");
    roiManager("Save", roiSavePath);
    print("rois saved");
    
    //save measurements
    roiManager("Measure");
    savePath = measurementFolder + images[i] + "_results.csv";
    saveAs("Results", savePath);
    print("results saved");
    
    run("Clear Results"); 
    //run("Save As...", "measurements=[" + measurementFolder + images[i] + "_measurements.csv]");
	run("Close All");
}

print("done");

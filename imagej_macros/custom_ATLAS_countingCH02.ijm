/*
This macro performs the following tasks:
1. Selects the ch2 images and performs a maximum Z-projection.
2. Allows drawing oval ROIs on the Z-projected ch2 image and saves them to the ROI Manager.
3. Measures the x, y coordinates (centroid), area, mean pixel value, min & max gray value, standard deviation, and modal gray value of the same ROIs across each color channel for the same brain slice.
4. Saves the measurement data with sufficient information to track which image the ROIs came from.
5. Repeats the process for each brain slice in batch mode.
*/

inputFolder = "/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS/exp89.animal2.VISp.ORB/";
//getDirectory("Choose Animal Directory");
outputFolder = "/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/";
//getDirectory("Choose the Parent Directory to Save Results");
print("Input Folder: " + inputFolder);
print("Output Folder: " + outputFolder);

resultsFolder = outputFolder + "/FIJI_Results/";
if (!File.isDirectory(resultsFolder)) {
    File.makeDirectory(resultsFolder);
}

images = getFileList(inputFolder);
setBatchMode(true);

// Loop through the images
for (i = 0; i < images.length; i++) {
    imageName = images[i];
    if (endsWith(imageName, "ch2.tif")) {
        inputPath = inputFolder + imageName;
        open(inputPath);
        print("opening: " + inputPath);
        waitForUser("manual open till fix..");
        imageTitle = getTitle();
        imageBaseName = File.getNameWithoutExtension(inputPath); // Get file name without .tif extension
        selectImage(imageBaseName+".tif");
        // Max Z-projection
        run("Z Project...", "projection=[Max Intensity]");
        
        // Select the oval tool
        setTool("oval");

        // Prompt the user to draw ROIs and save them to the ROI Manager
        waitForUser("Draw oval ROIs and click OK to continue.");

        // Save ROIs to the ROI Manager
        run("ROI Manager...");
        roiManager("reset");
        roiManager("Add");

        // Save ROIs to a temporary file
        tempRoiPath = resultsFolder + imageBaseName + "_rois.zip";
        roiManager("Save", tempRoiPath);

        // Close the Z-projected image
        selectWindow(imageTitle);
        close();

        // Measure the ROIs in each color channel
        for (ch = 0; ch < 4; ch++) {
            chImagePath = replace(inputPath, "ch2", "ch" + ch);
            if (File.exists(chImagePath)) {
                open(chImagePath);
                run("Set Measurements...", "area standard centroid mean min max modal display redirect=None decimal=3");
                roiManager("Open", tempRoiPath);
                roiManager("Select", 0);
                roiManager("Measure");

                // Ensure Results table is visible
                if (isResultsWindowVisible()) {
                    // Save the measurements
                    resultsPath = resultsFolder + imageBaseName + "_ch" + ch + "_results.csv";
                    saveAs("Results", resultsPath);
                    run("Clear Results");
                } else {
                    print("Error: Results table not found for " + imageBaseName + "_ch" + ch);
                }

                // Close the image
                selectWindow(getTitle());
                close();
            }
        }
        // Delete the temporary ROI file
        File.delete(tempRoiPath);
    }
}

// Function to check if the Results window is visible
function isResultsWindowVisible() {
    result = false;
    windows = getList("window.titles");
    for (i = 0; i < windows.length; i++) {
        if (windows[i] == "Results") {
            result = true;
            break;
        }
    }
    return result;
}

print("Done");

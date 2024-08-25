import groovy.io.FileType
import java.awt.image.BufferedImage
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.gui.commands.ProjectCommands
import qupath.lib.images.ImageData.ImageType
import qupath.lib.color.ColorDeconvolutionStains
import static qupath.lib.scripting.QP.runPlugin
import qupath.lib.gui.tools.MeasurementExporter
import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.algorithms.IntensityFeaturesPlugin
import qupath.lib.analysis.DelaunayTools

// Define paths for input images and output project
// TODO: change to arguments
images_path = args[0]
output_project_path = args[1]

// Create File objects for input and output directories
images_dir = new File(images_path)
project_dir = new File(output_project_path + File.separator + "QPProject")

// Create the project directory if it doesn't exist
if (!project_dir.exists()) {
    project_dir.mkdirs()
    println("New project created at: " + project_dir.getCanonicalPath())
} else {
    println("Project already exists at: " + project_dir.getCanonicalPath())
}

// Create a new QuPath project
def project = Projects.createProject(project_dir, BufferedImage.class)

// Initialize an empty list to store image files
def files = []

// Recursively search for .svs files in the input directory
// TODO: add optional argument to set file type
images_dir.eachFileRecurse(FileType.FILES) { file ->
    if (file.getName().toLowerCase().endsWith("."+args[3])) {
        files << file
        println("Added: " + file.getCanonicalPath())
    }
}

// Add each image to the project
for (file in files) {
    println(file)
    def imagePath = file.getCanonicalPath()
    def support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, imagePath)
    println(support)
    println(new File(imagePath).exists())
    def builder = support.builders.get(0)
    def originalName = file.getName().take(file.getName().lastIndexOf('.'))
    print originalName
    def entry = project.addImage(builder)
    entry.setImageName(originalName)
}
project.syncChanges()

// Process each image in the project
for (entry in project.getImageList()) {   // Loop through all images in the project
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()
    def name = entry.getImageName()
    def roi = ROIs.createRectangleROI(0, 0, imageData.getServer().getWidth(), imageData.getServer().getHeight(), ImagePlane.getDefaultPlane())

    // Create a ROI covering the entire image
    imageData.setImageType(ImageType.BRIGHTFIELD_H_E)
    def stains = ColorDeconvolutionStains.parseColorDeconvolutionStainsArg(
        '{"Name" : "H&E default",'+
        ' "Stain 1" : "Hematoxylin",'+
        ' "Values 1" : "0.65111 0.70119 0.29049",'+
        ' "Stain 2" : "Eosin",'+
        ' "Values 2" : "0.2159 0.8012 0.5581",'+
        ' "Background" : " 255 255 255"}')

    // Set image type and color deconvolution stains
    imageData.setColorDeconvolutionStains(stains)

    // Create an annotation object for the entire image
    def annotation = PathObjects.createAnnotationObject(roi)
    hierarchy.addObject(annotation)
    hierarchy.getSelectionModel().setSelectedObject(annotation)
    def cell = PathObjects.createDetectionObject(roi)
    createSelectAllObject(true)

    // TODO: Make configurable using a yaml file for example.
    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', imageData, args[2])
    runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin',imageData, '{"fwhmMicrons":20.0,"smoothWithinClasses":false}')
    runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin',imageData, '{"fwhmMicrons":50.0,"smoothWithinClasses":false}')
    runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin',imageData, '{"fwhmMicrons":100.0,"smoothWithinClasses":false}')
    selectAllObjects(hierarchy, true)
    runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin',imageData, '{"pixelSizeMicrons":1.0,"region":"ROI","tileSizeMicrons":25.0,"colorOD":true,"colorStain1":true,"colorStain2":true,"colorStain3":true,"colorRed":true,"colorGreen":true,"colorBlue":true,"colorHue":true,"colorSaturation":true,"colorBrightness":true,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":true,"haralickDistance":1,"haralickBins":32}')
    hierarchy.getSelectionModel().setSelectedObject(annotation)
    runPlugin('qupath.opencv.features.DelaunayClusteringPlugin',imageData, '{"distanceThresholdMicrons":0.0,"limitByClass":false,"addClusterMeasurements":true}')


    // Save detection measurements
    def filename = entry.getImageName() + '.tsv'
    saveDetectionMeasurements(imageData, project_dir.getCanonicalPath() + File.separator + filename)
}


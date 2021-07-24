const faceapi = require('@vladmandic/face-api');
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const jpeg = require('jpeg-js');
// If 0 is a total match
const maxDescriptorDistance = 0.6;
//Load faceapi networks
const faceDetectionLandmark68 = faceapi.nets.faceLandmark68Net;
const faceRecognition = faceapi.nets.faceRecognitionNet;
const ssdMobilenetv1 = faceapi.nets.ssdMobilenetv1;
const MODELS = './models';
//const faceDetectionOptions = new faceapi;

async function detectFaces(img) {
  //Decode image buffer to JPEG
  let imgJpeg = jpeg.decode(img, true)
  //Create a tensorflow object
  let tFrame = tf.browser.fromPixels(imgJpeg)
  //Detect all faces
  let fullFaceDescriptions = await faceapi.detectAllFaces(tFrame).withFaceLandmarks().withFaceDescriptors();
  fullFaceDescriptions = faceapi.resizeResults(fullFaceDescriptions, imgJpeg)
  return fullFaceDescriptions;
}

async function detectFace(fullFaceDescriptions, label, fromPath) {
  //If matches stops loooping to next file
  let match;
  // Compare folder
  let compare = './images/' + label + '/';
  // Read example folder 
  fs.readdir(compare, async (_err, files) => {
    // For each file detect the face
    files.forEach(async (file) => {
      let imgPath = compare + file
      fs.readFile(imgPath, async (_err, img) => {
        //If matches stops loooping to next file
        if(match) {
          return;
        }
        //Decode image buffer to JPEG
        let imgJpeg = jpeg.decode(img, true)
        //Create a tensorflow object
        let tFrame = tf.browser.fromPixels(imgJpeg)
        //Detect face
        const fullFaceDescription = await faceapi.detectSingleFace(tFrame).withFaceLandmarks().withFaceDescriptor()
        if (!fullFaceDescription) {
          return;
        }
        //Generarate face description and labels, can add more folders or image templates to create match
        const faceDescriptors = [fullFaceDescription.descriptor]
        let labeledFaceDescriptors = new faceapi.LabeledFaceDescriptors(label, faceDescriptors)

        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance)
        // find best match from generated labels
        const results = fullFaceDescriptions.map(element => {
          return faceMatcher.findBestMatch(element.descriptor);
        });
        
        results.forEach((bestMatch) => {
          //Ignore uknown faces
          if (bestMatch.label != 'unknown')
            //Face found
            console.log("Found " + bestMatch.label + " on " + imgPath + " from " + fromPath, bestMatch.distance);
            match = true;
            return;
        });
      });
    });
  });
}

async function process() {
  // Load models
  await faceDetectionLandmark68.loadFromDisk(MODELS);
  await faceRecognition.loadFromDisk(MODELS);
  await ssdMobilenetv1.loadFromDisk(MODELS);

  // Samples folder
  let faces = './images/samples/';
  // Read example folder 
  fs.readdir(faces, async (_err, files) => {
    // For each file detect the face
    files.forEach(async (file, index) => {
      let imgPath = faces + file
      fs.readFile(imgPath, async (_err, img) => {
        //Read all detected faces
        let fullFaceDescriptions = await detectFaces(img, imgPath, index)
        //Find single label faces, comparing from compare folder, in this case has only Hilton faces
        detectFace(fullFaceDescriptions, "Hilton", imgPath)

      });
    });
  });

}
//Call Process after 100ms
setTimeout(process, 100);
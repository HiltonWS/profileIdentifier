const faceapi = require('@vladmandic/face-api');
const cv = require('opencv4nodejs');
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const jpeg = require('jpeg-js');
const childProcess = require("child_process");
const process = require('process');

// If 0 is a total match
const maxDescriptorDistance = 0.6;
const faceDetectionLandmark68 = faceapi.nets.faceLandmark68Net;
const faceRecognition = faceapi.nets.faceRecognitionNet;
const ssdMobilenetv1 = faceapi.nets.ssdMobilenetv1;
const MODELS = './models';

let globalFrame = null;
let lastFrame = null;

async function processFrame(capture, videoCapture) {
  if (globalFrame == null || lastFrame == globalFrame){
      return;
  } 
  
  lastFrame = globalFrame;
  let tFrame = faceapi.tf.tensor3d(globalFrame, [480, 640, 3])
  //From camera
  const fullFaceDescriptions = await faceapi.detectAllFaces(tFrame).withFaceLandmarks().withFaceDescriptors()
  if (!fullFaceDescriptions) {
    return;
  }
  //Search from face samples in detected label
  await detectFace(fullFaceDescriptions, "Hilton", videoCapture);
  // New frame
  await capture();

}

async function detectFace(fullFaceDescriptions, label, videoCapture) {
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
        if (match) {
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
            if(bestMatch.label === "Hilton"){
              //Open google chrome in linux, can  be change to detect operating system
              let child = childProcess.spawn('google-chrome', ["--restore-last-session", "--profile-directory=Default"],
               { 
                 detached: true,
                 stdio: [ 'ignore', 'ignore', 'ignore' ]
               });
              child.unref();
              videoCapture.release();
              process.exit(1);
            }
            
          match = true;
          return;
        });
      });
    });
  });
}

function closeIfChromeIsRunning() { 
  //Search to chrome on process, can  be change to detect operating system
  childProcess.exec("pgrep chrome", (error, stdout, stderr) => {
    //If is no chrome return
    if (error || stderr) {
        return;
    }
    //If chrome found show message and exit exit
    if(stdout){
      console.log("\nChrome is runnning")
      process.exit(1);
    }
  });
}


async function run() {
  //Loading models
  await faceDetectionLandmark68.loadFromDisk(MODELS);
  await faceRecognition.loadFromDisk(MODELS);
  await ssdMobilenetv1.loadFromDisk(MODELS);
  //Attach a video device
  let videoCapture = new cv.VideoCapture(0);

  //Capturing
  let capture = async () => {
    let frame = videoCapture.read();
    //If has a frame with a width
    if (frame.cols > 0) {
      //Turn into a buffered data
      let data = new Uint8Array(frame.getData().buffer);
      //Used in processframe
      globalFrame = data;

    }

    setTimeout(async () => {
      //Star process frames
      await processFrame(capture, videoCapture);
    }, 0);

  }
  await capture();

}

closeIfChromeIsRunning();
run()

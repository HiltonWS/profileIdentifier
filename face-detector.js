const faceapi = require('@vladmandic/face-api');
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const jpeg = require('jpeg-js');
const sharp = require('sharp');

const minConfidence = 0.5;
//Use the algorithm tiny face detector with a confidence more then 50%
const faceDetectionNet = faceapi.nets.tinyFaceDetector;
const faceDetectionOptions = new faceapi.TinyFaceDetectorOptions({ minConfidence });

async function detectFaces(img, imgPath, index) {
  //Decode image buffer to JPEG
  let imgJpeg = jpeg.decode(img, true)
  //Create a tensorflow object
  let tFrame = tf.browser.fromPixels(imgJpeg)
  //Detect all faces
  let faces = await faceapi.detectAllFaces(tFrame, faceDetectionOptions)
  if (faces.length > 0) {
    //For each face cut and save it
    faces.forEach(async (face) => {
      let box = face.box
      saveFace(imgPath, box, index)
    })
    return;
  }
}

async function saveFace(path, box, suffix) {
  //Lets define the params of face region
  let left = Math.round(Math.abs(box.left))
  let top = Math.round(box.top)
  let width = Math.round(box.width)
  let height = Math.round(box.height)
  let imgPath = './images/faces/face' + suffix + '.jpg';
  let size = 150;
  let region = {
    left: left,
    top: top,
    width: width,
    height: height
  }
  //If all params is ok, and the face are is valid try extract
  try {
    await sharp(path).extract(region).resize(size, size).greyscale().toFile(imgPath)
  //When error try resize
  } catch {
    await sharp(path).resize(region).resize(size, size).greyscale().toFile(imgPath)
  }
}

async function process() {
  // Load models from disk (in this case we only use tinyfacedetector model)
  await faceDetectionNet.loadFromDisk('./models');
  // Samples folder
  let samples = './images/samples/'
  // Reade example folder 
  fs.readdir(samples, async (_err, files) => {
    // For each file detect the face
    files.forEach(async (file, index) => {
      let imgPath = samples + file
      fs.readFile(imgPath, async (_err, img) => {
        await detectFaces(img, imgPath, index)
      });
    });
  });
}
//Call Process after 100ms
setTimeout(process, 100);
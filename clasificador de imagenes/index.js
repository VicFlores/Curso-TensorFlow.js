let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');
const webcamEl = document.getElementById('webcam');
/* const classifier = knnClassifier.create(); */

async function app() {
  net = await mobilenet.load();
  var result = await net.classify(imgEl);

  displayImagePrediction();

  let webcam = await tf.data.webcam(webcamEl);

  while (true) {
    const img = await webcam.capture();

    console.log(img);

    const result = await net.classify(img);

    document.getElementById('console').innerHTML =
      'prediction: ' +
      result[0].className +
      'probability: ' +
      result[0].probability;

    img.dispose();

    await tf.nextFrame();
  }
}

imgEl.onload = async function () {
  displayImagePrediction();
};

async function addExample(classId) {}

async function displayImagePrediction() {
  try {
    result = await net.classify(imgEl);
    descEl.innerHTML = JSON.stringify(result);
  } catch (error) {
    console.log(error);
  }
}

count = 0;
async function cambiarImagen() {
  count = count + 1;
  imgEl.src = 'https://picsum.photos/200/300?random=' + count;
}

app();

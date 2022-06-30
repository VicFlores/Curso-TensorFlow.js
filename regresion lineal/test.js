var modelo;
var stopTraining;

async function getData() {
  const response = await fetch(
    'https://static.platzi.com/media/public/uploads/datos-entrenamiento_15cd99ce-3561-494e-8f56-9492d4e86438.json'
  );
  const dataHouses = await response.json();
  let cleanDataHouses = dataHouses.map((house) => ({
    precio: house.Precio,
    cuartos: house.NumeroDeCuartosPromedio,
  }));

  cleanDataHouses = cleanDataHouses.filter(
    (house) => house.precio != null && house.cuartos != null
  );

  return cleanDataHouses;
}

function viewData(data) {
  const values = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }));

  tfvis.render.scatterplot(
    { name: 'Cuartos vs Precio' },
    { values },
    {
      xLabel: 'Rooms',
      yLabel: 'Prices',
      height: 300,
    }
  );
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

function convertDataToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const entries = data.map((d) => d.cuartos);
    const label = data.map((d) => d.precio);

    const tensorEntries = tf.tensor2d(entries, [entries.length, 1]);
    const tensorLabel = tf.tensor2d(entries, [label.length, 1]);

    const entriesMax = tensorEntries.max();
    const entriesMin = tensorEntries.min();
    const labelsMax = tensorEntries.max();
    const labelsMin = tensorEntries.min();

    const entriesNormalized = tensorEntries
      .sub(entriesMin)
      .div(entriesMax)
      .sub(entriesMin);

    const labelNormalized = tensorLabel
      .sub(labelsMin)
      .div(labelsMax)
      .sub(labelsMin);

    return {
      entries: entriesNormalized,
      label: labelNormalized,
      entriesMax,
      entriesMin,
      labelsMax,
      labelsMin,
    };
  });
}

const optimizator = tf.train.adam();
const losses = tf.losses.meanSquaredError;
const metric = ['mse'];

async function trainingModel(model, inputs, labels) {
  model.compile({
    optimizer: optimizator,
    loss: losses,
    metrics: metric,
  });

  const surface = { name: 'show.history live', tab: 'Training' };

  const sizeBatch = 28;

  const epochs = 50;

  const history = [];

  return await model.fit(inputs, labels, {
    sizeBatch,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        history.push(log);
        tfvis.show.history(surface, history, ['loss', 'mse']);

        if (stopTraining) {
          modelo.stopTraining = true;
        }
      },
    },
  });
}

async function saveModel() {
  const saveResultModel = await modelo.save('downloads://modelo-regresion');
}

async function loadModel() {
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightInput = document.getElementById('upload-weights');

  mode = await tf.loadLayersModel(
    tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightInput.files[0]])
  );
  console.log('Modelo cargado');
}

async function verCurvaInferencia() {
  var data = await getData();
  var tensorData = await convertDataToTensor(data);
  const { entriesMax, entriesMin, labelsMax, labelsMin } = tensorData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);

    const preds = modelo.predict(xs.reshape([100, 1]));

    const desnormX = xs.mul(entriesMax.sub(entriesMin)).add(entriesMin);

    const desnormY = preds.mul(labelsMax.sub(labelsMin)).add(labelsMin);

    return [desnormX.dataSync(), desnormY.dataSync()];
  });

  const pointsPrediction = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originPoints = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }));

  tfvis.render.scatterplot(
    { name: 'Predicciones vs Originales' },
    {
      values: [originPoints, pointsPrediction],
      series: ['originales', 'predicciones'],
    },
    {
      xLabel: 'Cuartos',
      yLabel: 'Precios',
      height: 300,
    }
  );
}

async function run() {
  const data = await getData();

  viewData(data);

  modelo = createModel();

  const tensorData = convertDataToTensor(data);
  const { entries, label } = tensorData;

  trainingModel(modelo, entries, label);
}

run();

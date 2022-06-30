var stopTraining;

// Obtenemos la informacion
async function getData() {
  const datosCasasR = await fetch(
    'https://static.platzi.com/media/public/uploads/datos-entrenamiento_15cd99ce-3561-494e-8f56-9492d4e86438.json'
  );
  const datosCasas = await datosCasasR.json();
  const datosLimpios = datosCasas
    .map((casa) => ({
      precio: casa.Precio,
      cuartos: casa.NumeroDeCuartosPromedio,
    }))
    .filter((casa) => casa.precio != null && casa.cuartos != null);

  return datosLimpios;
}

// Nos ayuda a visualizar los datos
function visualizarDatos(data) {
  const valores = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }));

  tfvis.render.scatterplot(
    { name: 'Cuartos vs Precio' },
    { values: valores },
    {
      xLabel: 'Cuartos',
      yLabel: 'Precio',
      height: 300,
    }
  );
}

// Creacion de modelo
function crearModelo() {
  const modelo = tf.sequential();

  // agregar capa oculta que va a recibir 1 dato
  modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // agregar una capa de salida que va a tener 1 sola unidad
  modelo.add(tf.layers.dense({ units: 1, useBias: true }));

  return modelo;
}

// Convertimos nuestros datos en tensores
function convertirDatosATensores(data) {
  // Nos permite esta function  deshacernos de todos los datos que no son utiles para nosotros
  return tf.tidy(() => {
    // mezclamos los datos de forma aleatoria
    tf.util.shuffle(data);

    // Mapeamos nuestra info
    const entradas = data.map((d) => d.cuartos);
    const etiquetas = data.map((d) => d.precio);

    // Tranformamos nuestras variables de datos en tensores 2d
    const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
    const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

    // Desregularizamos las variables de datos minimas y maximas
    const entradasMax = tensorEntradas.max();
    const entradasMin = tensorEntradas.min();
    const etiquetasMax = tensorEtiquetas.max();
    const etiquetasMin = tensorEtiquetas.min();

    // Creamos las entradas normalizadas, (dato-min) / (max-min)
    const entradasNormalizadas = tensorEntradas
      .sub(entradasMin)
      .div(entradasMax.sub(entradasMin));
    const etiquetasNormalizadas = tensorEtiquetas
      .sub(etiquetasMin)
      .div(etiquetasMax.sub(etiquetasMin));

    return {
      entradas: entradasNormalizadas,
      etiquetas: etiquetasNormalizadas,
      entradasMax,
      entradasMin,
      etiquetasMax,
      etiquetasMin,
    };
  });
}

const optimizador = tf.train.adam();
const funcion_perdida = tf.losses.meanSquaredError;
const metricas = ['mse']; // mse => meanSquaredError

async function entrenarModelo(model, inputs, labels) {
  // Preperamos el modelo para su entrenamiento
  model.compile({
    optimizer: optimizador,
    loss: funcion_perdida,
    metrics: metricas,
  });

  // Desplegamos la forma en que nuestro modelo entrena y como va optimizando y reduciendo el error
  const surface = { name: 'show.history live', tab: 'Training' };

  // Definimos el numero de registros que incluira dentro del entrenamiento
  const tamanioBatch = 28;

  // Definimos la cantidad de vueltas que queremos que le de al modelo
  const epochs = 50;

  // Nos permite mantener las metricas en el proceso de entrenamiento para poder graficarlas
  const history = [];

  return await model.fit(inputs, labels, {
    tamanioBatch,
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

// Guardamos nuestro modelo ya entrenado
async function guardarModelo() {
  const saveResult = await modelo.save('downloads://modelo-regresion');
}

// Cargamos nuestro modelo
async function cargarModelo() {
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');

  modelo = await tf.loadLayersModel(
    tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]])
  );
  console.log('Modelo Cargado');
}

// Mostramos la curva de inferencia()
async function verCurvaInferencia() {
  var data = await getData();
  var tensorData = await convertirDatosATensores(data);

  const { entradasMax, entradasMin, etiquetasMin, etiquetasMax } = tensorData;

  const [xs, preds] = tf.tidy(() => {
    // designamos los datos de entrada
    const xs = tf.linspace(0, 1, 100);

    // una vez entrenado nuestro modelo, usamos su prediction
    const preds = modelo.predict(xs.reshape([100, 1]));

    // aplicamos la desnormalizacion en X,Y
    const desnormX = xs.mul(entradasMax.sub(entradasMin)).add(entradasMin);

    const desnormY = preds
      .mul(etiquetasMax.sub(etiquetasMin))
      .add(etiquetasMin);

    return [desnormX.dataSync(), desnormY.dataSync()];
  });

  const puntosPrediccion = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const puntosOriginales = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }));

  tfvis.render.scatterplot(
    { name: 'Prediccion vs Originales' },
    {
      values: [puntosOriginales, puntosPrediccion],
      series: ['originales', 'predicciones'],
    },
    {
      xLabel: 'Cuartos',
      yLabel: 'Precio',
      height: 300,
    }
  );
}

var modelo;

async function run() {
  // Utilizamos los datos previamente cargados
  const data = await getData();

  // Visualizamos los datos designados
  visualizarDatos(data);

  // Implementamos el modelo
  modelo = crearModelo();

  // Traemos los datos convertidos en tensores
  const tensorData = convertirDatosATensores(data);
  const { entradas, etiquetas } = tensorData;

  // Comenzamos a entrenar nuestros modelos
  await entrenarModelo(modelo, entradas, etiquetas);
}

run();

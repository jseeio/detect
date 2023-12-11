const outputs = document.getElementById('outputs')

const pre = document.createElement('pre')
pre.setAttribute('id', 'log')
pre.setAttribute('style', 'overflow: scroll; height: 100px;')
outputs.appendChild(pre)

const videoElement = document.createElement('video')
videoElement.setAttribute('width', '320')
videoElement.setAttribute('height', '240')
videoElement.setAttribute('autoplay', '')
videoElement.setAttribute('style', 'display: none;')
videoElement.setAttribute('id', 'video')
outputs.appendChild(videoElement)

const canvasElement = document.createElement('canvas')
canvasElement.setAttribute('width', '320')
canvasElement.setAttribute('height', '240')
canvasElement.setAttribute('style', 'border:1px solid #FF0000;')
canvasElement.setAttribute('id', 'canvas')
outputs.appendChild(canvasElement)

const outputElement = document.createElement('div')
outputElement.setAttribute('id', 'output')
outputs.appendChild(outputElement)

const log = (...a) => {
    console.log(...a)
    pre.innerHTML += `\n${a.join(' ')}`
    pre.scrollTop = pre.scrollHeight // auto scroll to bottom
}

// const MODEL_URL = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tf_model.h5'
const MODEL_URL = 'http://localhost:8080/tmp/clip.h5'
const VOCAB_URL = 'https://cdn.jsdelivr.net/npm/modelzoo@latest/bpe_simple_vocab_16e6.txt'

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

class Detector {
    constructor() {
        this.tokenizer = null
        this.model = null
    }

    async load() {
        const h5 = await h5wasm.ready
        const { FS } = await h5.ready


        log('Building tokenizer from:', VOCAB_URL)
        let r1 = await fetch(VOCAB_URL)
        let vocab = await r1.text()
        this.tokenizer = new modelzoo.tokenizers.SimpleTokenizer(vocab);

        // Try loading from IndexedDB
        try {
            log('Loading model from IndexedDB')
            const modelVisual = await tf.loadLayersModel('indexeddb://clip-vit-base-patch32-visual')
            const modelText = await tf.loadLayersModel('indexeddb://clip-vit-base-patch32-text')
            const modelSimilarity = await tf.loadLayersModel('indexeddb://clip-vit-base-patch32-similarity')
            const config = {
              "patchSize": 32,
              "inputResolution": 224,
              "gridSize": 7,
              "visionWidth": 768,
              "visionLayers": 12,
              "blockSize": 77,
              "nEmbd": 512,
              "nHead": 8,
              "vocabSize": 49408,
              "nLayer": 12,
              "debug": false,
              "joint": false
            }
            this.model = new modelzoo.models.CLIP(config, modelVisual, modelText, modelSimilarity)
            log('Loaded model from IndexedDB')
        } catch (e) {
            // Try loading from URL
            log('Failed to load model from IndexedDB')
            log('Loading from:', MODEL_URL)
            log('This may take a couple of minutes...')
            let r2 = await fetch(MODEL_URL)
            let ab = await r2.arrayBuffer()
            FS.writeFile("clip.h5", new Uint8Array(ab))
            let f = new h5wasm.File("clip.h5", "r")
            this.model = await modelzoo.models.CLIP.buildModel(f, 'clip-vit-base-h5', async (stats) => {
                log(`[${stats["i"]}/${stats["total"]}] Setting ${stats["name"]} with shape: [${stats["shape"]}]`)
            });

            try {
                log('Storing model in IndexedDB')
                log('Stored visual model in IndexedDB')
                await this.model.visual.save('indexeddb://clip-vit-base-patch32-visual')
                log('Stored text model in IndexedDB')
                await this.model.text.save('indexeddb://clip-vit-base-patch32-text')
                log('Stored similarity model in IndexedDB')
                await this.model.similarity.save('indexeddb://clip-vit-base-patch32-similarity')
            } catch (e) {
                log('Failed to store model in IndexedDB:', e)
            }
        }
    }

    async detect(params) {
        const model = this.model
        const classes = params.classes.split(',').map(x => x.trim())
        const tokens = tf.tensor(this.tokenizer.tokenize(classes))
        const embText = model.encodeText(tokens);

        const ctx = canvasElement.getContext('2d');

        // Initialize webcam
        if (navigator.mediaDevices.getUserMedia) {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true })
            videoElement.srcObject = stream;
        }
        await new Promise(resolve => setTimeout(resolve, 1000))
        async function predict() {
            ctx.drawImage(videoElement, 0, 0, 320, 240);
            await new Promise(resolve => setTimeout(resolve, 100))
            const img = ctx.getImageData(0, 0, 320, 240);
            const imgTensor = tf.browser.fromPixels(img, 3);
            const embVis = model.encodeImage(imgTensor);
            const [logitsPerImage, logitsPerText] = model.similarity.apply([embVis, embText]);
            const logitsVis = await tf.squeeze(logitsPerImage).array()

//             if (clogitsVisData['2'] > clogitsVisData['1'] && clogitsVisData['2'] > clogitsVisData['0']) {
//                 document.body.style.backgroundColor = 'blue';
//             } else if (clogitsVisData['0'] > clogitsVisData['1']) {
//                 document.body.style.backgroundColor = 'red';
//             } else {
//                 document.body.style.backgroundColor = 'green';
//             }
            outputElement.innerHTML = ''
            classes.forEach((x, i) => {
                const div = document.createElement('div')
                div.innerHTML = `${x}: ${logitsVis[i]}`
                outputElement.appendChild(div)
            })
            await new Promise(resolve => setTimeout(resolve, 1000))
            await predict()
        }
        await predict()
    }

    async run(params) {
        switch (params.caller) {
            case 'load':
                return await this.load(params)
            case 'run':
                if (!this.model)
                    await this.load()
                return await this.detect(params)
            default:
                throw new Error('Unknown caller')
        }

    }
}
"use strict";

const fs = require("fs");
const path = require("path");
const { parentPort, workerData } = require("worker_threads");

const ROOT = workerData.root;
const DATEN = workerData.daten;
const GELERNT = workerData.gelernt;
const OPTIONS = workerData.options || {};

const MODEL_FILE = path.join(GELERNT, "model.json");
const TOKENIZER_FILE = path.join(GELERNT, "tokenizer.json");
const CONFIG_FILE = path.join(GELERNT, "config.json");
const STATE_FILE = path.join(GELERNT, "training-state.json");

let stopRequested = false;

if (parentPort) {
    parentPort.on("message", message => {
        if (message && message.type === "stop") {
            stopRequested = true;
        }
    });
}


/* =========================================================
   HILFSFUNKTIONEN
   ========================================================= */

function send(type, data = {}) {
    if (parentPort) {
        parentPort.postMessage({
            type,
            ...data
        });
    }
}

function atomicWrite(file, data) {
    const temp =
        file +
        ".tmp-" +
        process.pid +
        "-" +
        Date.now();

    fs.writeFileSync(
        temp,
        data,
        "utf8"
    );

    fs.renameSync(
        temp,
        file
    );
}

function readJSON(file) {
    return JSON.parse(
        fs.readFileSync(
            file,
            "utf8"
        )
    );
}

function sleep(ms) {
    return new Promise(
        resolve => setTimeout(resolve, ms)
    );
}


/* =========================================================
   DATEN
   ========================================================= */

const QUESTION_KEYS = [
    "frage",
    "question",
    "user",
    "input",
    "prompt"
];

const ANSWER_KEYS = [
    "antwort",
    "answer",
    "assistant",
    "output",
    "response"
];

function findKey(object, keys) {
    if (
        !object ||
        typeof object !== "object"
    ) {
        return null;
    }

    for (const key of Object.keys(object)) {
        if (
            keys.includes(
                key.toLowerCase()
            )
        ) {
            return key;
        }
    }

    return null;
}

function collectExamples(value, output) {
    if (Array.isArray(value)) {
        for (const item of value) {
            collectExamples(
                item,
                output
            );
        }

        return;
    }

    if (
        !value ||
        typeof value !== "object"
    ) {
        return;
    }

    const questionKey =
        findKey(
            value,
            QUESTION_KEYS
        );

    const answerKey =
        findKey(
            value,
            ANSWER_KEYS
        );

    if (
        questionKey &&
        answerKey &&
        typeof value[questionKey] === "string" &&
        typeof value[answerKey] === "string"
    ) {
        const question =
            value[questionKey].trim();

        const answer =
            value[answerKey].trim();

        if (
            question &&
            answer
        ) {
            output.push({
                question,
                answer
            });
        }
    }

    for (const key of Object.keys(value)) {
        collectExamples(
            value[key],
            output
        );
    }
}

function collectTexts(value, output) {
    if (typeof value === "string") {
        const text = value.trim();

        if (text) {
            output.push(text);
        }

        return;
    }

    if (Array.isArray(value)) {
        for (const item of value) {
            collectTexts(
                item,
                output
            );
        }

        return;
    }

    if (
        value &&
        typeof value === "object"
    ) {
        for (const key of Object.keys(value)) {
            collectTexts(
                value[key],
                output
            );
        }
    }
}

function getDataFiles() {
    if (!fs.existsSync(DATEN)) {
        fs.mkdirSync(
            DATEN,
            {
                recursive: true
            }
        );

        return [];
    }

    return fs.readdirSync(
        DATEN,
        {
            withFileTypes: true
        }
    )
    .filter(
        entry =>
            entry.isFile() &&
            entry.name
                .toLowerCase()
                .endsWith(".json")
    )
    .map(
        entry =>
            path.join(
                DATEN,
                entry.name
            )
    );
}

function loadTrainingData() {
    const files =
        getDataFiles();

    const examples = [];
    const texts = [];
    const seen = new Set();

    for (const file of files) {
        if (stopRequested) {
            break;
        }

        let raw;

        try {
            raw =
                fs.readFileSync(
                    file,
                    "utf8"
                );
        } catch (error) {
            console.error(
                "Fehler beim Lesen:",
                file,
                error.message
            );

            continue;
        }

        try {
            const json =
                JSON.parse(raw);

            const localExamples = [];

            collectExamples(
                json,
                localExamples
            );

            for (
                const example of
                localExamples
            ) {
                const id =
                    example.question +
                    "\n" +
                    example.answer;

                if (!seen.has(id)) {
                    seen.add(id);

                    examples.push(
                        example
                    );
                }
            }

            collectTexts(
                json,
                texts
            );

        } catch {
            if (raw.trim()) {
                texts.push(
                    raw.trim()
                );
            }
        }
    }

    /*
     * Falls die JSON-Dateien keine
     * Frage/Antwort-Struktur besitzen.
     */

    if (
        examples.length === 0 &&
        texts.length > 0
    ) {
        for (const text of texts) {
            if (
                text.length >= 4
            ) {
                examples.push({
                    question:
                        "Erzähle etwas über dieses Thema.",
                    answer:
                        text
                });
            }
        }
    }

    return {
        files,
        examples,
        texts
    };
}


/* =========================================================
   TOKENIZER
   ========================================================= */

function loadTokenizerClass() {
    const file =
        path.join(
            ROOT,
            "tokenizer.js"
        );

    if (!fs.existsSync(file)) {
        throw new Error(
            "tokenizer.js fehlt."
        );
    }

    const module =
        require(file);

    const Tokenizer =
        module.AdvancedTokenizer ||
        module;

    if (
        typeof Tokenizer !==
        "function"
    ) {
        throw new Error(
            "AdvancedTokenizer konnte nicht geladen werden."
        );
    }

    return Tokenizer;
}

function createTokenizer() {
    const Tokenizer =
        loadTokenizerClass();

    let tokenizer = null;

    /*
     * Bereits trainierter Tokenizer.
     */

    if (
        fs.existsSync(
            TOKENIZER_FILE
        )
    ) {
        try {
            const saved =
                readJSON(
                    TOKENIZER_FILE
                );

            if (
                typeof Tokenizer.fromJSON ===
                "function"
            ) {
                tokenizer =
                    Tokenizer.fromJSON(
                        saved
                    );
            } else {
                tokenizer =
                    new Tokenizer();

                if (
                    typeof tokenizer.import ===
                    "function"
                ) {
                    tokenizer.import(
                        saved
                    );
                }
            }
        } catch (error) {
            console.warn(
                "Gespeicherter Tokenizer konnte nicht geladen werden:",
                error.message
            );
        }
    }

    if (!tokenizer) {
        tokenizer =
            new Tokenizer({
                vocabSize:
                    OPTIONS.vocabSize ||
                    8192,

                minFrequency:
                    OPTIONS.minFrequency ||
                    2,

                maxTokenLength:
                    OPTIONS.maxTokenLength ||
                    64,

                maxMerges:
                    OPTIONS.maxMerges ||
                    16000
            });
    }

    return tokenizer;
}


/* =========================================================
   TOKENIZER TRAINIEREN
   ========================================================= */

async function trainTokenizer(
    tokenizer,
    examples
) {
    if (
        stopRequested
    ) {
        return tokenizer;
    }

    /*
     * Falls tokenizer.train existiert,
     * wird der gesamte Text verwendet.
     */

    if (
        typeof tokenizer.train ===
        "function"
    ) {
        const texts =
            examples.map(
                example =>
                    "<|user|>\n" +
                    example.question +
                    "\n<|end|>\n" +
                    "<|assistant|>\n" +
                    example.answer +
                    "\n<|end|>"
            );

        try {
            await tokenizer.train(
                texts
            );
        } catch (error) {
            /*
             * Manche Versionen erwarten
             * andere Parameter.
             *
             * Der bereits vorhandene
             * Tokenizer bleibt trotzdem
             * verwendbar.
             */
            console.warn(
                "Tokenizer-Training:",
                error.message
            );
        }
    }

    return tokenizer;
}


/* =========================================================
   TOKENIZER SPEICHERN
   ========================================================= */

function exportTokenizer(
    tokenizer
) {
    let data = null;

    if (
        typeof tokenizer.export ===
        "function"
    ) {
        data =
            tokenizer.export();
    } else if (
        typeof tokenizer.toJSON ===
        "function"
    ) {
        data =
            tokenizer.toJSON();
    } else {
        data =
            tokenizer;
    }

    atomicWrite(
        TOKENIZER_FILE,
        JSON.stringify(
            data,
            null,
            2
        )
    );
}


/* =========================================================
   MODELL
   ========================================================= */

function loadModelClass() {
    const file =
        path.join(
            ROOT,
            "model.js"
        );

    if (!fs.existsSync(file)) {
        throw new Error(
            "model.js fehlt."
        );
    }

    const module =
        require(file);

    const Model =
        module.LanguageModel ||
        module.LargeLanguageModel;

    if (
        typeof Model !==
        "function"
    ) {
        throw new Error(
            "LanguageModel konnte nicht geladen werden."
        );
    }

    return Model;
}


/* =========================================================
   MODELL-KONFIGURATION
   ========================================================= */

function createModelConfig(
    tokenizer
) {
    const vocabSize =
        Number(
            tokenizer.vocabSize ||
            tokenizer.vocab?.length ||
            OPTIONS.vocabSize ||
            8192
        );

    /*
     * Standardmäßig ein kleineres Modell,
     * damit reines JavaScript auf einem
     * normalen Windows-PC überhaupt
     * sinnvoll starten kann.
     */

    return {
        vocabSize,

        contextSize:
            Number(
                OPTIONS.contextSize ||
                256
            ),

        embeddingSize:
            Number(
                OPTIONS.embeddingSize ||
                192
            ),

        layers:
            Number(
                OPTIONS.layers ||
                6
            ),

        heads:
            Number(
                OPTIONS.heads ||
                6
            ),

        headSize:
            Number(
                OPTIONS.headSize ||
                32
            ),

        feedForwardSize:
            Number(
                OPTIONS.feedForwardSize ||
                512
            ),

        dropout: 0,

        rmsEpsilon:
            0.00001,

        learningRate:
            Number(
                OPTIONS.learningRate ||
                0.0003
            ),

        beta1:
            0.9,

        beta2:
            0.95,

        weightDecay:
            0.01,

        gradientClip:
            1.0,

        temperature:
            0.85,

        topK:
            40,

        topP:
            0.92,

        repetitionPenalty:
            1.08
    };
}


/* =========================================================
   MODELL ERSTELLEN / LADEN
   ========================================================= */

function createModel(
    tokenizer
) {
    const Model =
        loadModelClass();

    let config =
        createModelConfig(
            tokenizer
        );

    /*
     * Bestehende Konfiguration
     * bevorzugen.
     */

    if (
        fs.existsSync(
            CONFIG_FILE
        )
    ) {
        try {
            const saved =
                readJSON(
                    CONFIG_FILE
                );

            if (
                saved.model
            ) {
                config =
                    Object.assign(
                        {},
                        config,
                        saved.model
                    );
            }

        } catch {}
    }

    /*
     * Vom Startbefehl übergebene
     * Werte überschreiben die Datei.
     */

    const overrideKeys = [
        "vocabSize",
        "contextSize",
        "embeddingSize",
        "layers",
        "heads",
        "headSize",
        "feedForwardSize",
        "learningRate",
        "gradientClip"
    ];

    for (
        const key of
        overrideKeys
    ) {
        if (
            OPTIONS[key] !==
            undefined
        ) {
            config[key] =
                OPTIONS[key];
        }
    }

    const model =
        new Model(
            config
        );

    /*
     * Bereits gelerntes Modell laden.
     */

    if (
        fs.existsSync(
            MODEL_FILE
        )
    ) {
        try {
            const saved =
                readJSON(
                    MODEL_FILE
                );

            if (
                typeof model.load ===
                "function"
            ) {
                model.load(
                    saved
                );
            }

        } catch (error) {
            console.warn(
                "Altes Modell konnte nicht geladen werden:",
                error.message
            );
        }
    }

    return {
        model,
        config
    };
}


/* =========================================================
   TRAININGSBEISPIELE
   ========================================================= */

function formatExample(
    example
) {
    return (
        "<|user|>\n" +
        example.question +
        "\n<|end|>\n" +
        "<|assistant|>\n" +
        example.answer +
        "\n<|end|>"
    );
}


function makeSequences(
    tokenizer,
    examples,
    sequenceLength
) {
    const sequences = [];

    for (
        const example of
        examples
    ) {
        if (
            stopRequested
        ) {
            break;
        }

        const text =
            formatExample(
                example
            );

        let tokens;

        try {
            tokens =
                tokenizer.encode(
                    text
                );
        } catch {
            continue;
        }

        if (
            !Array.isArray(tokens) &&
            !(tokens instanceof Uint32Array)
        ) {
            continue;
        }

        const list =
            Array.from(
                tokens
            );

        if (
            list.length < 2
        ) {
            continue;
        }

        if (
            list.length <=
            sequenceLength
        ) {
            sequences.push(
                list
            );

            continue;
        }

        /*
         * Lange Beispiele werden
         * in mehrere Abschnitte geteilt.
         */

        for (
            let start = 0;
            start < list.length;
            start += sequenceLength - 16
        ) {
            if (
                stopRequested
            ) {
                break;
            }

            const part =
                list.slice(
                    start,
                    start +
                    sequenceLength
                );

            if (
                part.length >= 2
            ) {
                sequences.push(
                    part
                );
            }
        }
    }

    return sequences;
}


/* =========================================================
   TRAINING
   ========================================================= */

async function trainModel(
    model,
    tokenizer,
    sequences,
    epochs
) {
    if (
        typeof model.trainStep !==
        "function" &&
        typeof model.trainBackprop !==
        "function"
    ) {
        throw new Error(
            "Das Modell besitzt keine trainierbare trainStep/trainBackprop-Funktion."
        );
    }

    let globalStep = 0;
    let lastLoss = null;

    const learningRate =
        Number(
            OPTIONS.learningRate ||
            model.config?.learningRate ||
            0.0003
        );

    if (
        model.config
    ) {
        model.config.learningRate =
            learningRate;
    }

    for (
        let epoch = 1;
        epoch <= epochs;
        epoch++
    ) {
        if (
            stopRequested
        ) {
            break;
        }

        /*
         * Zufällige Reihenfolge.
         */

        const order =
            sequences
                .map(
                    (_, index) =>
                        index
                );

        for (
            let i =
                order.length - 1;
            i > 0;
            i--
        ) {
            const j =
                Math.floor(
                    Math.random() *
                    (i + 1)
                );

            [
                order[i],
                order[j]
            ] =
            [
                order[j],
                order[i]
            ];
        }

        for (
            let position = 0;
            position <
            order.length;
            position++
        ) {
            if (
                stopRequested
            ) {
                break;
            }

            const sequence =
                sequences[
                    order[position]
                ];

            if (
                sequence.length < 2
            ) {
                continue;
            }

            const input =
                sequence.slice(
                    0,
                    -1
                );

            const target =
                sequence.slice(
                    1
                );

            try {
                let result;

                if (
                    typeof model.trainStep ===
                    "function"
                ) {
                    result =
                        model.trainStep(
                            input,
                            target,
                            {
                                learningRate
                            }
                        );

                } else {
                    result =
                        model.trainBackprop(
                            input,
                            target,
                            {
                                learningRate
                            }
                        );
                }

                if (
                    typeof result ===
                    "number"
                ) {
                    lastLoss =
                        result;

                } else if (
                    result &&
                    typeof result.loss ===
                    "number"
                ) {
                    lastLoss =
                        result.loss;
                }

            } catch (error) {
                console.warn(
                    "Trainingsschritt:",
                    error.message
                );
            }

            globalStep++;


            /*
             * Fortschritt regelmäßig an
             * server.js schicken.
             */

            if (
                globalStep % 5 === 0
            ) {
                send(
                    "progress",
                    {
                        epoch,
                        step:
                            globalStep,
                        loss:
                            lastLoss
                    }
                );

                /*
                 * Worker kurz Luft geben.
                 */
                await sleep(0);
            }
        }

        /*
         * Nach jeder Epoche speichern.
         */

        saveModel(
            model,
            tokenizer,
            {
                epoch,
                step:
                    globalStep,
                loss:
                    lastLoss
            }
        );

        send(
            "progress",
            {
                epoch,
                step:
                    globalStep,
                loss:
                    lastLoss
            }
        );
    }

    return {
        epoch:
            Math.min(
                epochs,
                Math.max(
                    0,
                    epochs
                )
            ),

        step:
            globalStep,

        loss:
            lastLoss
    };
}


/* =========================================================
   MODELL SPEICHERN
   ========================================================= */

function serializeModel(
    model
) {
    if (
        typeof model.toJSON ===
        "function"
    ) {
        return model.toJSON();
    }

    if (
        typeof model.serialize ===
        "function"
    ) {
        return model.serialize();
    }

    if (
        typeof model.export ===
        "function"
    ) {
        return model.export();
    }

    throw new Error(
        "Das Modell besitzt keine JSON-Serialisierung."
    );
}


function saveModel(
    model,
    tokenizer,
    state
) {
    fs.mkdirSync(
        GELERNT,
        {
            recursive:
                true
        }
    );


    /*
     * Modell
     */

    const modelData =
        serializeModel(
            model
        );

    atomicWrite(
        MODEL_FILE,
        JSON.stringify(
            modelData
        )
    );


    /*
     * Tokenizer
     */

    exportTokenizer(
        tokenizer
    );


    /*
     * Konfiguration
     */

    const config =
        {
            version: 1,

            updatedAt:
                new Date()
                    .toISOString(),

            model:
                model.config ||
                {},

            training: {
                learningRate:
                    OPTIONS.learningRate ||
                    model.config?.learningRate ||
                    0.0003,

                contextSize:
                    OPTIONS.contextSize ||
                    model.config?.contextSize ||
                    256
            }
        };


    atomicWrite(
        CONFIG_FILE,
        JSON.stringify(
            config,
            null,
            2
        )
    );


    /*
     * Trainingszustand
     */

    const trainingState =
        {
            version: 1,

            epoch:
                state.epoch,

            step:
                state.step,

            loss:
                state.loss,

            sourceFolder:
                DATEN,

            sourceFiles:
                getDataFiles()
                    .map(
                        file =>
                            path.basename(
                                file
                            )
                    ),

            updatedAt:
                new Date()
                    .toISOString()
        };


    atomicWrite(
        STATE_FILE,
        JSON.stringify(
            trainingState,
            null,
            2
        )
    );
}


/* =========================================================
   HAUPTPROGRAMM
   ========================================================= */

async function main() {
    try {
        fs.mkdirSync(
            GELERNT,
            {
                recursive:
                    true
            }
        );


        send(
            "progress",
            {
                phase:
                    "daten",
                epoch:
                    0,
                step:
                    0,
                loss:
                    null
            }
        );


        const dataset =
            loadTrainingData();


        if (
            stopRequested
        ) {
            send(
                "finished",
                {
                    stopped:
                        true
                }
            );

            return;
        }


        if (
            dataset.examples.length ===
            0
        ) {
            throw new Error(
                "Keine Trainingsdaten in DATEN gefunden."
            );
        }


        send(
            "progress",
            {
                phase:
                    "tokenizer",
                examples:
                    dataset.examples.length
            }
        );


        const tokenizer =
            createTokenizer();


        await trainTokenizer(
            tokenizer,
            dataset.examples
        );


        exportTokenizer(
            tokenizer
        );


        if (
            stopRequested
        ) {
            send(
                "finished",
                {
                    stopped:
                        true
                }
            );

            return;
        }


        send(
            "progress",
            {
                phase:
                    "sequenzen"
            }
        );


        const sequenceLength =
            Number(
                OPTIONS.sequenceLength ||
                256
            );


        const sequences =
            makeSequences(
                tokenizer,
                dataset.examples,
                sequenceLength
            );


        if (
            sequences.length ===
            0
        ) {
            throw new Error(
                "Es konnten keine Trainingssequenzen erzeugt werden."
            );
        }


        send(
            "progress",
            {
                phase:
                    "modell",
                sequences:
                    sequences.length
            }
        );


        const {
            model,
            config
        } =
        createModel(
            tokenizer
        );


        /*
         * Config sofort sichern,
         * damit das Modell beim nächsten
         * Start dieselbe Architektur besitzt.
         */

        atomicWrite(
            CONFIG_FILE,
            JSON.stringify(
                {
                    version: 1,

                    updatedAt:
                        new Date()
                            .toISOString(),

                    model:
                        config,

                    training: {
                        learningRate:
                            config.learningRate,

                        sequenceLength
                    }
                },
                null,
                2
            )
        );


        const epochs =
            Math.max(
                1,
                Number(
                    OPTIONS.epochs ||
                    10
                )
            );


        send(
            "progress",
            {
                phase:
                    "training",
                epoch:
                    0,
                step:
                    0,
                loss:
                    null,
                totalEpochs:
                    epochs,
                sequences:
                    sequences.length
            }
        );


        const result =
            await trainModel(
                model,
                tokenizer,
                sequences,
                epochs
            );


        /*
         * Auch nach einem Stop
         * letzten Zustand speichern.
         */

        saveModel(
            model,
            tokenizer,
            result
        );


        send(
            "finished",
            {
                stopped:
                    stopRequested,

                epoch:
                    result.epoch,

                step:
                    result.step,

                loss:
                    result.loss
            }
        );

    } catch (error) {

        console.error(
            "TRAIN-WORKER FEHLER:",
            error
        );

        send(
            "error",
            {
                error:
                    error.message,

                stack:
                    error.stack
            }
        );
    }
}


main();

"use strict";

/*
================================================================
  train-worker.js
  LUMORA

  - lädt ALLE JSON-Dateien aus DATEN/
  - durchsucht JSON rekursiv nach Frage/Antwort
  - entfernt doppelte Beispiele
  - verwendet den LUMORA-Tokenizer
  - verwendet die tatsächliche Tokenizer-Vokabulargröße
  - kleines Modell für Render
  - speichert Modell + Tokenizer + Config
================================================================
*/

const fs = require("fs");
const path = require("path");
const {
    parentPort,
    workerData
} = require("worker_threads");


/* =========================================================
   PFADE
   ========================================================= */

const ROOT =
    workerData.root || __dirname;

const DATEN =
    workerData.daten ||
    path.join(
        ROOT,
        "DATEN"
    );

const GELERNT =
    workerData.gelernt ||
    path.join(
        ROOT,
        "GELERNT"
    );

const OPTIONS =
    workerData.options || {};


const MODEL_FILE =
    path.join(
        GELERNT,
        "model.json"
    );

const TOKENIZER_FILE =
    path.join(
        GELERNT,
        "tokenizer.json"
    );

const CONFIG_FILE =
    path.join(
        GELERNT,
        "config.json"
    );

const STATE_FILE =
    path.join(
        GELERNT,
        "training-state.json"
    );


/* =========================================================
   STOP
   ========================================================= */

let stopRequested = false;

if (parentPort) {

    parentPort.on(
        "message",
        message => {

            if (
                message &&
                (
                    message.type === "stop" ||
                    message.action === "stop"
                )
            ) {
                stopRequested = true;
            }

        }
    );

}


/* =========================================================
   KOMMUNIKATION
   ========================================================= */

function send(
    type,
    data = {}
) {

    if (!parentPort) {
        return;
    }

    parentPort.postMessage({
        type,
        ...data
    });

}


/* =========================================================
   JSON
   ========================================================= */

function writeJSON(
    file,
    data
) {

    fs.mkdirSync(
        path.dirname(file),
        {
            recursive: true
        }
    );

    const temp =
        file + ".tmp";

    fs.writeFileSync(
        temp,
        JSON.stringify(
            data,
            null,
            2
        ),
        "utf8"
    );

    fs.renameSync(
        temp,
        file
    );

}


function readJSON(
    file
) {

    return JSON.parse(
        fs.readFileSync(
            file,
            "utf8"
        )
    );

}


/* =========================================================
   DATEN
   ========================================================= */

function getDataFiles() {

    if (
        !fs.existsSync(
            DATEN
        )
    ) {

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
            entry.name
    )

    .sort(
        (a, b) =>
            a.localeCompare(
                b,
                "de",
                {
                    numeric: true,
                    sensitivity: "base"
                }
            )
    );

}


/* =========================================================
   TRAININGSDATEN LADEN
   ========================================================= */

function loadTrainingExamples() {

    const files =
        getDataFiles();

    const examples = [];


    const questionKeys = [
        "frage",
        "question",
        "user",
        "input",
        "prompt"
    ];


    const answerKeys = [
        "antwort",
        "answer",
        "assistant",
        "output",
        "response"
    ];


    function scan(
        value
    ) {

        if (stopRequested) {
            return;
        }


        if (
            Array.isArray(
                value
            )
        ) {

            for (
                const item of value
            ) {
                scan(item);
            }

            return;
        }


        if (
            !value ||
            typeof value !== "object"
        ) {
            return;
        }


        let question = null;
        let answer = null;


        for (
            const key of
            Object.keys(value)
        ) {

            const lower =
                key.toLowerCase();


            if (
                questionKeys.includes(
                    lower
                )
            ) {
                question =
                    value[key];
            }


            if (
                answerKeys.includes(
                    lower
                )
            ) {
                answer =
                    value[key];
            }

        }


        if (
            typeof question === "string" &&
            typeof answer === "string" &&
            question.trim() &&
            answer.trim()
        ) {

            examples.push({

                question:
                    question.trim(),

                answer:
                    answer.trim()

            });

        }


        for (
            const key of
            Object.keys(value)
        ) {

            scan(
                value[key]
            );

        }

    }


    for (
        const filename of files
    ) {

        if (stopRequested) {
            break;
        }


        const file =
            path.join(
                DATEN,
                filename
            );


        let raw;

        try {

            raw =
                fs.readFileSync(
                    file,
                    "utf8"
                );

        } catch (error) {

            console.error(
                "Datei konnte nicht gelesen werden:",
                filename,
                error.message
            );

            continue;

        }


        try {

            const parsed =
                JSON.parse(
                    raw
                );

            scan(parsed);

        } catch (error) {

            console.error(
                "JSON konnte nicht verarbeitet werden:",
                filename,
                error.message
            );

        }

    }


    /*
     * Doppelte Beispiele entfernen.
     */

    const unique = [];
    const seen = new Set();


    for (
        const example of examples
    ) {

        const id =
            example.question +
            "\n" +
            example.answer;


        if (
            !seen.has(id)
        ) {

            seen.add(id);

            unique.push(
                example
            );

        }

    }


    return {

        files,

        examples:
            unique

    };

}


/* =========================================================
   TOKENIZER
   ========================================================= */

function loadTokenizer() {

    const tokenizerFile =
        path.join(
            ROOT,
            "tokenizer.js"
        );


    if (
        !fs.existsSync(
            tokenizerFile
        )
    ) {

        throw new Error(
            "tokenizer.js wurde nicht gefunden."
        );

    }


    delete require.cache[
        require.resolve(
            tokenizerFile
        )
    ];


    const loaded =
        require(
            tokenizerFile
        );


    const AdvancedTokenizer =
        loaded.AdvancedTokenizer ||
        loaded.default ||
        loaded;


    if (
        AdvancedTokenizer &&
        typeof AdvancedTokenizer.create ===
        "function"
    ) {

        return AdvancedTokenizer.create();

    }


    if (
        typeof AdvancedTokenizer ===
        "function"
    ) {

        return new AdvancedTokenizer();

    }


    throw new Error(
        "AdvancedTokenizer konnte nicht erstellt werden."
    );

}


/* =========================================================
   TOKENIZER TRAINING
   ========================================================= */

async function trainTokenizerSafe(
    tokenizer,
    examples
) {

    send(
        "progress",
        {

            phase:
                "tokenizer",

            progress:
                10,

            message:
                "Tokenizer wird vorbereitet..."

        }
    );


    if (stopRequested) {
        return tokenizer;
    }


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


    /*
     * Bei kleinen Datensätzen kein BPE.
     * Der Tokenizer verwendet dann seine
     * Zeichen-/Byte-Fallbacks.
     */

    if (
        typeof tokenizer.trainBPE ===
        "function"
    ) {

        if (
            texts.length >= 100
        ) {

            send(
                "progress",
                {

                    phase:
                        "tokenizer",

                    progress:
                        20,

                    message:
                        "BPE-Tokenizer wird trainiert..."

                }
            );


            try {

                await Promise.race([

                    Promise.resolve(
                        tokenizer.trainBPE(
                            texts
                        )
                    ),

                    new Promise(
                        (_, reject) => {

                            setTimeout(
                                () => {

                                    reject(
                                        new Error(
                                            "BPE-Training Timeout"
                                        )
                                    );

                                },
                                30000
                            );

                        }
                    )

                ]);


                send(
                    "progress",
                    {

                        phase:
                            "tokenizer",

                        progress:
                            70,

                        message:
                            "BPE-Training abgeschlossen."

                    }
                );


            } catch (error) {

                console.warn(
                    "BPE übersprungen:",
                    error.message
                );


                send(
                    "progress",
                    {

                        phase:
                            "tokenizer",

                        progress:
                            70,

                        message:
                            "BPE fehlgeschlagen – vorhandener Tokenizer wird verwendet."

                    }
                );

            }

        } else {

            send(
                "progress",
                {

                    phase:
                        "tokenizer",

                    progress:
                        70,

                    message:
                        "Kleiner Datensatz – BPE wird übersprungen."

                }
            );

        }

    }


    send(
        "progress",
        {

            phase:
                "tokenizer",

            progress:
                100,

            message:
                "Tokenizer bereit."

        }
    );


    return tokenizer;

}


/* =========================================================
   TOKENIZER SPEICHERN
   ========================================================= */

function exportTokenizer(
    tokenizer
) {

    let data;


    if (
        typeof tokenizer.export ===
        "function"
    ) {

        data =
            tokenizer.export();

    }

    else if (
        typeof tokenizer.toJSON ===
        "function"
    ) {

        data =
            tokenizer.toJSON();

    }

    else {

        throw new Error(
            "Tokenizer kann nicht exportiert werden."
        );

    }


    writeJSON(
        TOKENIZER_FILE,
        data
    );

}


/* =========================================================
   ECHTE VOKABULARGRÖSSE
   ========================================================= */

function getTokenizerVocabSize(
    tokenizer
) {

    /*
     * Das ist wichtig:
     *
     * Der LUMORA-Tokenizer besitzt keine
     * feste tokenizer.vocabSize-Eigenschaft.
     *
     * Die tatsächliche Größe steht in
     * tokenizer.vocabulary.length.
     */

    if (
        tokenizer &&
        Array.isArray(
            tokenizer.vocabulary
        )
    ) {

        return tokenizer.vocabulary.length;

    }


    if (
        tokenizer &&
        tokenizer.idToToken instanceof Map
    ) {

        return tokenizer.idToToken.size;

    }


    throw new Error(
        "Die tatsächliche Tokenizer-Vokabulargröße konnte nicht ermittelt werden."
    );

}


/* =========================================================
   MODELL
   ========================================================= */

function loadModelClass() {

    const modelFile =
        path.join(
            ROOT,
            "model.js"
        );


    if (
        !fs.existsSync(
            modelFile
        )
    ) {

        throw new Error(
            "model.js wurde nicht gefunden."
        );

    }


    delete require.cache[
        require.resolve(
            modelFile
        )
    ];


    const loaded =
        require(
            modelFile
        );


    const Model =
        loaded.LanguageModel ||
        loaded.default ||
        loaded;


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


function createModel(
    tokenizer
) {

    const Model =
        loadModelClass();


    /*
     * WICHTIG:
     * Niemals wieder blind 8192 verwenden.
     */

    const vocabSize =
        getTokenizerVocabSize(
            tokenizer
        );


    /*
     * Kleines Modell für Render.
     *
     * embeddingSize =
     * heads * headSize
     *
     * 32 = 1 * 32
     */

    const config = {

        vocabSize,

        contextSize:
            Math.max(
                8,
                Math.min(
                    128,
                    Number(
                        OPTIONS.contextSize ??
                        16
                    )
                )
            ),

        embeddingSize:
            Math.max(
                32,
                Number(
                    OPTIONS.embeddingSize ??
                    32
                )
            ),

        layers:
            Math.max(
                1,
                Math.min(
                    4,
                    Number(
                        OPTIONS.layers ??
                        1
                    )
                )
            ),

        heads:
            Math.max(
                1,
                Math.min(
                    4,
                    Number(
                        OPTIONS.heads ??
                        1
                    )
                )
            ),

        headSize:
            Math.max(
                8,
                Math.min(
                    64,
                    Number(
                        OPTIONS.headSize ??
                        32
                    )
                )
            ),

        feedForwardSize:
            Math.max(
                32,
                Math.min(
                    256,
                    Number(
                        OPTIONS.feedForwardSize ??
                        64
                    )
                )
            ),

        dropout:
            0,

        rmsEpsilon:
            0.00001,

        learningRate:
            Number(
                OPTIONS.learningRate ??
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
            0.8,

        topK:
            20,

        topP:
            0.9,

        repetitionPenalty:
            1.08
    };


    /*
     * Sicherheit:
     * embeddingSize muss heads * headSize sein.
     */

    config.embeddingSize =
        config.heads *
        config.headSize;


    const model =
        new Model(
            config
        );


    return {

        model,

        config

    };

}


/* =========================================================
   TRAININGSFORMAT
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


/* =========================================================
   TRAININGSSEQUENZEN
   ========================================================= */

function createSequences(
    tokenizer,
    examples,
    sequenceLength
) {

    const sequences = [];


    for (
        const example of examples
    ) {

        if (stopRequested) {
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

        } catch (error) {

            console.warn(
                "Encode-Fehler:",
                error.message
            );

            continue;

        }


        if (
            !Array.isArray(
                tokens
            )
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


        /*
         * Kurze Sequenz.
         */

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
         * Lange Sequenz aufteilen.
         */

        for (
            let start = 0;
            start < list.length;
            start += sequenceLength
        ) {

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
        "Modell besitzt keine Exportfunktion."
    );

}


function saveEverything(
    model,
    tokenizer,
    config,
    state
) {

    fs.mkdirSync(
        GELERNT,
        {
            recursive: true
        }
    );


    writeJSON(
        MODEL_FILE,
        serializeModel(
            model
        )
    );


    exportTokenizer(
        tokenizer
    );


    writeJSON(
        CONFIG_FILE,
        {

            version:
                2,

            name:
                "LUMORA",

            updatedAt:
                new Date().toISOString(),

            model:
                config,

            training: {

                epochs:
                    Math.max(
                        1,
                        Number(
                            OPTIONS.epochs ??
                            1
                        )
                    ),

                sequenceLength:
                    Math.max(
                        2,
                        Number(
                            OPTIONS.sequenceLength ??
                            16
                        )
                    ),

                learningRate:
                    Number(
                        OPTIONS.learningRate ??
                        0.0003
                    )

            }

        }
    );


    writeJSON(
        STATE_FILE,
        {

            version:
                2,

            epoch:
                state.epoch,

            step:
                state.step,

            loss:
                state.loss,

            examples:
                state.examples,

            sequences:
                state.sequences,

            updatedAt:
                new Date().toISOString()

        }
    );

}


/* =========================================================
   TRAINING
   ========================================================= */

async function train(
    model,
    sequences,
    config
) {

    const epochs =
        Math.max(
            1,
            Number(
                OPTIONS.epochs ??
                1
            )
        );


    let step = 0;
    let loss = null;


    for (
        let epoch = 1;
        epoch <= epochs;
        epoch++
    ) {

        if (stopRequested) {
            break;
        }


        for (
            let index = 0;
            index < sequences.length;
            index++
        ) {

            if (stopRequested) {
                break;
            }


            const sequence =
                sequences[index];


            if (
                sequence.length < 2
            ) {

                continue;

            }


            try {

                let result;


                /*
                 * Das aktuelle model.js
                 * verwendet trainStep(tokens).
                 *
                 * Deshalb wird die komplette
                 * Sequenz übergeben.
                 */

                if (
                    typeof model.trainStep ===
                    "function"
                ) {

                    result =
                        model.trainStep(
                            sequence
                        );

                }

                else if (
                    typeof model.trainBackprop ===
                    "function"
                ) {

                    result =
                        model.trainBackprop(
                            sequence
                        );

                }

                else {

                    throw new Error(
                        "trainStep/trainBackprop fehlt."
                    );

                }


                if (
                    typeof result ===
                    "number"
                ) {

                    loss =
                        result;

                }

                else if (
                    result &&
                    typeof result.loss ===
                    "number"
                ) {

                    loss =
                        result.loss;

                }

            } catch (error) {

                console.warn(
                    "Trainingsschritt fehlgeschlagen:",
                    error.message
                );

            }


            step++;


            const progress =
                Math.round(

                    (

                        (
                            epoch - 1
                        ) /

                        epochs

                    )

                    +

                    (

                        (
                            index + 1
                        ) /

                        sequences.length /

                        epochs

                    )

                ) * 100;


            send(
                "progress",
                {

                    phase:
                        "training",

                    progress:
                        Math.min(
                            99,
                            progress
                        ),

                    epoch,

                    step,

                    totalEpochs:
                        epochs,

                    loss,

                    message:
                        "Training: Epoche " +
                        epoch +
                        "/" +
                        epochs

                }
            );


            /*
             * Render etwas Luft geben.
             */

            if (
                step % 5 ===
                0
            ) {

                await new Promise(
                    resolve =>
                        setImmediate(
                            resolve
                        )
                );

            }

        }


        /*
         * Zwischenstand speichern.
         */

        saveEverything(
            model,
            nullSafeTokenizer,
            config,
            {
                epoch,
                step,
                loss,
                examples:
                    0,
                sequences:
                    sequences.length
            }
        );


        send(
            "epoch",
            {

                epoch,

                step,

                loss,

                totalEpochs:
                    epochs

            }
        );

    }


    return {

        epoch:
            stopRequested
                ? Math.max(
                    0,
                    Math.min(
                        epochs,
                        epochs - 1
                    )
                )
                : epochs,

        step,

        loss

    };

}


/*
 * Wird nur verwendet, damit der Zwischenstand
 * nicht mit einem undefined-Tokenizer abstürzt.
 */
let nullSafeTokenizer = null;


/* =========================================================
   START
   ========================================================= */

async function main() {

    try {

        fs.mkdirSync(
            GELERNT,
            {
                recursive: true
            }
        );


        send(
            "started",
            {

                phase:
                    "daten",

                message:
                    "Trainingsdaten werden geladen..."

            }
        );


        const dataset =
            loadTrainingExamples();


        if (
            dataset.examples.length ===
            0
        ) {

            throw new Error(
                "Keine Trainingsdaten gefunden."
            );

        }


        send(
            "progress",
            {

                phase:
                    "daten",

                progress:
                    5,

                examples:
                    dataset.examples.length,

                files:
                    dataset.files,

                message:
                    dataset.examples.length +
                    " Trainingsbeispiele aus " +
                    dataset.files.length +
                    " JSON-Datei(en) gefunden."

            }
        );


        if (stopRequested) {

            send(
                "stopped",
                {
                    message:
                        "Training gestoppt."
                }
            );

            return;

        }


        /* =================================================
           TOKENIZER
           ================================================= */

        const tokenizer =
            loadTokenizer();


        nullSafeTokenizer =
            tokenizer;


        await trainTokenizerSafe(
            tokenizer,
            dataset.examples
        );


        exportTokenizer(
            tokenizer
        );


        if (stopRequested) {

            send(
                "stopped",
                {
                    message:
                        "Training gestoppt."
                }
            );

            return;

        }


        /* =================================================
           VOKABULARUM
           ================================================= */

        const vocabSize =
            getTokenizerVocabSize(
                tokenizer
            );


        send(
            "progress",
            {

                phase:
                    "tokenizer",

                progress:
                    100,

                vocabSize,

                message:
                    "Tokenizer bereit: " +
                    vocabSize +
                    " Token."

            }
        );


        /* =================================================
           SEQUENZEN
           ================================================= */

        send(
            "progress",
            {

                phase:
                    "sequenzen",

                progress:
                    0,

                message:
                    "Trainingssequenzen werden erstellt..."

            }
        );


        const sequenceLength =
            Math.max(
                2,
                Number(
                    OPTIONS.sequenceLength ??
                    16
                )
            );


        const sequences =
            createSequences(
                tokenizer,
                dataset.examples,
                sequenceLength
            );


        if (
            sequences.length ===
            0
        ) {

            throw new Error(
                "Keine Trainingssequenzen erzeugt."
            );

        }


        send(
            "progress",
            {

                phase:
                    "sequenzen",

                progress:
                    100,

                sequences:
                    sequences.length,

                message:
                    sequences.length +
                    " Trainingssequenzen erstellt."

            }
        );


        /* =================================================
           MODELL
           ================================================= */

        send(
            "progress",
            {

                phase:
                    "modell",

                progress:
                    0,

                message:
                    "LUMORA-Modell wird erstellt..."

            }
        );


        const {
            model,
            config
        } =
            createModel(
                tokenizer
            );


        send(
            "progress",
            {

                phase:
                    "modell",

                progress:
                    100,

                vocabSize:
                    config.vocabSize,

                message:
                    "LUMORA-Modell bereit."

            }
        );


        /* =================================================
           TRAINING
           ================================================= */

        send(
            "progress",
            {

                phase:
                    "training",

                progress:
                    0,

                epoch:
                    0,

                step:
                    0,

                message:
                    "LUMORA-Training gestartet..."

            }
        );


        const result =
            await train(
                model,
                sequences,
                config
            );


        /* =================================================
           FINAL SPEICHERN
           ================================================= */

        saveEverything(
            model,
            tokenizer,
            config,
            {

                epoch:
                    result.epoch,

                step:
                    result.step,

                loss:
                    result.loss,

                examples:
                    dataset.examples.length,

                sequences:
                    sequences.length

            }
        );


        if (stopRequested) {

            send(
                "stopped",
                {

                    epoch:
                        result.epoch,

                    step:
                        result.step,

                    loss:
                        result.loss,

                    message:
                        "Training gestoppt. LUMORA wurde gespeichert."

                }
            );

        }

        else {

            send(
                "finished",
                {

                    epoch:
                        result.epoch,

                    step:
                        result.step,

                    loss:
                        result.loss,

                    progress:
                        100,

                    vocabSize:
                        config.vocabSize,

                    examples:
                        dataset.examples.length,

                    sequences:
                        sequences.length,

                    message:
                        "LUMORA vollständig trainiert und gespeichert."

                }
            );

        }

    } catch (error) {

        console.error(
            "LUMORA TRAIN-WORKER ERROR:",
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

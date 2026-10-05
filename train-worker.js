"use strict";

const fs = require("fs");
const path = require("path");
const { parentPort, workerData } = require("worker_threads");

const ROOT =
    workerData.root ||
    __dirname;

const DATEN =
    workerData.daten ||
    path.join(ROOT, "DATEN");

const GELERNT =
    workerData.gelernt ||
    path.join(ROOT, "GELERNT");

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

let stopRequested = false;


/* =========================================================
   STOP
========================================================= */

if (parentPort) {
    parentPort.on(
        "message",
        message => {

            if (
                message &&
                (
                    message.type ===
                    "stop" ||
                    message.action ===
                    "stop"
                )
            ) {
                stopRequested = true;
            }
        }
    );
}


/* =========================================================
   MESSAGE
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
   DATEI
========================================================= */

function writeJSON(
    file,
    data
) {

    fs.mkdirSync(
        path.dirname(file),
        {
            recursive:
                true
        }
    );

    const temp =
        file +
        ".tmp";

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
                recursive:
                    true
            }
        );

        return [];
    }

    const result = [];


    function scanDirectory(
        directory
    ) {

        let entries;

        try {

            entries =
                fs.readdirSync(
                    directory,
                    {
                        withFileTypes:
                            true
                    }
                );

        } catch (error) {

            console.error(
                "[DATEN] Ordner konnte nicht gelesen werden:",
                directory,
                error.message
            );

            return;
        }


        for (
            const entry of
            entries
        ) {

            const fullPath =
                path.join(
                    directory,
                    entry.name
                );


            if (
                entry.isDirectory()
            ) {

                scanDirectory(
                    fullPath
                );

                continue;
            }


            if (
                entry.isFile() &&
                entry.name
                    .toLowerCase()
                    .endsWith(".json")
            ) {

                result.push(
                    fullPath
                );
            }
        }
    }


    scanDirectory(
        DATEN
    );


    result.sort(
        (a, b) =>
            a.localeCompare(
                b,
                "de",
                {
                    numeric:
                        true,
                    sensitivity:
                        "base"
                }
            )
    );


    return result;
}


/* =========================================================
   JSON → FRAGE / ANTWORT
========================================================= */

function extractExamples(
    value,
    output = []
) {

    if (
        value === null ||
        value === undefined
    ) {

        return output;
    }


    if (
        Array.isArray(value)
    ) {

        for (
            const item of
            value
        ) {

            if (stopRequested) {
                return output;
            }

            extractExamples(
                item,
                output
            );
        }

        return output;
    }


    if (
        typeof value !==
        "object"
    ) {

        return output;
    }


    const questionKeys = [
        "frage",
        "question",
        "prompt",
        "input",
        "user"
    ];


    const answerKeys = [
        "antwort",
        "answer",
        "response",
        "output",
        "assistant"
    ];


    let question = null;
    let answer = null;


    for (
        const key of
        Object.keys(value)
    ) {

        const lower =
            key
                .toLowerCase()
                .trim();


        if (
            questionKeys.includes(
                lower
            )
        ) {

            if (
                typeof value[key] ===
                "string"
            ) {

                question =
                    value[key];
            }
        }


        if (
            answerKeys.includes(
                lower
            )
        ) {

            if (
                typeof value[key] ===
                "string"
            ) {

                answer =
                    value[key];
            }
        }
    }


    if (
        typeof question ===
            "string" &&
        typeof answer ===
            "string"
    ) {

        const q =
            question
                .replace(
                    /\s+/g,
                    " "
                )
                .trim();


        const a =
            answer
                .trim();


        if (
            q.length > 0 &&
            a.length > 0
        ) {

            output.push({
                question:
                    q,
                answer:
                    a
            });
        }
    }


    for (
        const key of
        Object.keys(value)
    ) {

        if (stopRequested) {
            break;
        }

        const child =
            value[key];


        if (
            child &&
            typeof child ===
                "object"
        ) {

            extractExamples(
                child,
                output
            );
        }
    }


    return output;
}


/* =========================================================
   TRAININGSDATEN LADEN
========================================================= */

function loadTrainingExamples() {

    const files =
        getDataFiles();

    const rawExamples =
        [];

    const validFiles =
        [];

    const failedFiles =
        [];


    for (
        const file of files
    ) {

        if (stopRequested) {
            break;
        }


        try {

            const raw =
                fs.readFileSync(
                    file,
                    "utf8"
                );


            if (
                !raw.trim()
            ) {

                continue;
            }


            const parsed =
                JSON.parse(
                    raw
                );


            const before =
                rawExamples.length;


            extractExamples(
                parsed,
                rawExamples
            );


            if (
                rawExamples.length >
                before
            ) {

                validFiles.push(
                    file
                );
            }


        } catch (error) {

            failedFiles.push({
                file,
                error:
                    error.message
            });


            console.error(
                "[DATEN] Fehler:",
                file,
                error.message
            );
        }
    }


    /*
      Duplikate entfernen
    */

    const examples =
        [];

    const seen =
        new Set();


    for (
        const example of
        rawExamples
    ) {

        const question =
            example.question
                .toLowerCase()
                .replace(
                    /\s+/g,
                    " "
                )
                .trim();


        const answer =
            example.answer
                .replace(
                    /\s+/g,
                    " "
                )
                .trim();


        const key =
            question +
            "\n" +
            answer;


        if (
            seen.has(key)
        ) {
            continue;
        }


        seen.add(key);


        examples.push({
            question:
                example.question,
            answer
        });
    }


    return {
        files,
        validFiles,
        failedFiles,
        examples
    };
}


/* =========================================================
   TOKENIZER
========================================================= */

function loadTokenizer() {

    const tokenizerModule =
        path.join(
            ROOT,
            "tokenizer.js"
        );


    if (
        !fs.existsSync(
            tokenizerModule
        )
    ) {

        throw new Error(
            "tokenizer.js wurde nicht gefunden."
        );
    }


    delete require.cache[
        require.resolve(
            tokenizerModule
        )
    ];


    const loaded =
        require(
            tokenizerModule
        );


    const AdvancedTokenizer =
        loaded.AdvancedTokenizer ||
        loaded.default ||
        loaded;


    if (
        AdvancedTokenizer &&
        typeof
            AdvancedTokenizer.create ===
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
   TOKENIZER EXISTIEREND LADEN
========================================================= */

function loadExistingTokenizer() {

    if (
        !fs.existsSync(
            TOKENIZER_FILE
        )
    ) {

        return null;
    }


    try {

        const tokenizer =
            loadTokenizer();


        const data =
            readJSON(
                TOKENIZER_FILE
            );


        if (
            typeof tokenizer.import ===
            "function"
        ) {

            tokenizer.import(
                data
            );

            return tokenizer;
        }


        if (
            typeof tokenizer.fromJSON ===
            "function"
        ) {

            return tokenizer.fromJSON(
                data
            );
        }


    } catch (error) {

        console.warn(
            "[TOKENIZER] Vorhandener Tokenizer konnte nicht geladen werden:",
            error.message
        );
    }


    return null;
}


/* =========================================================
   TOKENIZER TRAINING
========================================================= */

async function prepareTokenizer(
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
                (
                    "<|user|>\n" +
                    example.question +
                    "\n<|end|>\n" +
                    "<|assistant|>\n" +
                    example.answer +
                    "\n<|end|>"
                )
        );


    /*
      Bestehenden Tokenizer behalten.
      Das verhindert, dass die Token-IDs bei jedem
      Training komplett verändert werden.
    */

    const existing =
        loadExistingTokenizer();


    if (existing) {

        send(
            "progress",
            {
                phase:
                    "tokenizer",

                progress:
                    100,

                message:
                    "Vorhandener Tokenizer wurde geladen."
            }
        );

        return existing;
    }


    /*
      Nur bei ausreichend Daten BPE trainieren.
      Bei kleinen Datenmengen ist der vorhandene
      Basis-Tokenizer besser.
    */

    if (
        texts.length >= 100 &&
        typeof tokenizer.trainBPE ===
            "function"
    ) {

        send(
            "progress",
            {
                phase:
                    "tokenizer",

                progress:
                    25,

                message:
                    "BPE-Tokenizer wird trainiert..."
            }
        );


        try {

            await tokenizer.trainBPE(
                texts
            );

        } catch (error) {

            console.warn(
                "[TOKENIZER] BPE fehlgeschlagen:",
                error.message
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
   TOKENIZER EXPORT
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

    } else if (
        typeof tokenizer.toJSON ===
        "function"
    ) {

        data =
            tokenizer.toJSON();

    } else {

        throw new Error(
            "Tokenizer besitzt keine Exportfunktion."
        );
    }


    writeJSON(
        TOKENIZER_FILE,
        data
    );
}


/* =========================================================
   VOCAB SIZE
========================================================= */

function getVocabSize(
    tokenizer
) {

    const candidates = [
        tokenizer?.vocabSize,
        tokenizer?.vocabulary?.length,
        tokenizer?.vocab?.length,
        tokenizer?.idToToken?.size
    ];


    for (
        const value of
        candidates
    ) {

        const n =
            Number(
                value
            );


        if (
            Number.isInteger(n) &&
            n > 0
        ) {

            return n;
        }
    }


    return 8192;
}


/* =========================================================
   MODELL LADEN
========================================================= */

function loadModelClass() {

    const modelModule =
        path.join(
            ROOT,
            "model.js"
        );


    if (
        !fs.existsSync(
            modelModule
        )
    ) {

        throw new Error(
            "model.js wurde nicht gefunden."
        );
    }


    delete require.cache[
        require.resolve(
            modelModule
        )
    ];


    const loaded =
        require(
            modelModule
        );


    const Model =
        loaded.LanguageModel ||
        loaded.LargeLanguageModel ||
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


/* =========================================================
   MODELL ERSTELLEN
========================================================= */

function createModel(
    tokenizer
) {

    const Model =
        loadModelClass();


    const vocabSize =
        getVocabSize(
            tokenizer
        );


    /*
      LUMORA LARGE

      Deutlich größer als die alte Version.
    */

    const config = {

        modelType:
            "LUMORA-LARGE",

        vocabSize,

        contextSize:
            Number(
                OPTIONS.contextSize ||
                256
            ),

        embeddingSize:
            Number(
                OPTIONS.embeddingSize ||
                128
            ),

        layers:
            Number(
                OPTIONS.layers ||
                4
            ),

        heads:
            Number(
                OPTIONS.heads ||
                4
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

        maxNgramOrder:
            8,

        maxNgramEntries:
            100000,

        learningRate:
            Number(
                OPTIONS.learningRate ||
                0.00025
            ),

        minLearningRate:
            0.00002,

        temperature:
            0.75,

        topK:
            40,

        topP:
            0.92,

        repetitionPenalty:
            1.08,

        seed:
            1337
    };


    /*
      Wichtig:
      Model.js selbst sorgt dafür,
      dass embeddingSize = heads * headSize ist.
    */

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
   ALTES MODELL LADEN
========================================================= */

function loadExistingModel(
    model
) {

    if (
        !fs.existsSync(
            MODEL_FILE
        )
    ) {

        return false;
    }


    try {

        const saved =
            readJSON(
                MODEL_FILE
            );


        /*
          Nur kompatible Modelle laden.

          Alte Versionen werden absichtlich
          nicht blind in das neue Modell geladen.
        */

        const savedConfig =
            saved?.config;


        const compatible =
            saved?.version >= 5 &&
            savedConfig &&
            Number(
                savedConfig.vocabSize
            ) ===
                Number(
                    model.config.vocabSize
                ) &&
            Number(
                savedConfig.contextSize
            ) ===
                Number(
                    model.config.contextSize
                ) &&
            Number(
                savedConfig.layers
            ) ===
                Number(
                    model.config.layers
                );


        if (
            !compatible
        ) {

            console.log(
                "[MODEL] Altes inkompatibles Modell wird nicht übernommen."
            );

            return false;
        }


        if (
            typeof model.load ===
            "function"
        ) {

            model.load(
                saved
            );

            console.log(
                "[MODEL] Vorhandenes kompatibles Modell geladen."
            );

            return true;
        }


    } catch (error) {

        console.warn(
            "[MODEL] Laden fehlgeschlagen:",
            error.message
        );
    }


    return false;
}


/* =========================================================
   TRAININGSTEXT
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
   SEQUENZEN ERSTELLEN
========================================================= */

function createSequences(
    tokenizer,
    examples,
    sequenceLength
) {

    const sequences =
        [];


    for (
        const example of
        examples
    ) {

        if (stopRequested) {
            break;
        }


        const text =
            formatExample(
                example
            );


        let encoded;


        try {

            encoded =
                tokenizer.encode(
                    text
                );

        } catch (error) {

            console.warn(
                "[ENCODE]",
                error.message
            );

            continue;
        }


        if (!encoded) {
            continue;
        }


        const tokens =
            Array.from(
                encoded
            )
                .map(Number)
                .filter(
                    id =>
                        Number.isInteger(
                            id
                        )
                );


        if (
            tokens.length < 2
        ) {
            continue;
        }


        /*
          Kurze Beispiele:
          direkt verwenden.
        */

        if (
            tokens.length <=
            sequenceLength
        ) {

            sequences.push(
                tokens
            );

            continue;
        }


        /*
          Lange Beispiele:
          über mehrere Fenster verteilen.
        */

        for (
            let start = 0;
            start <
                tokens.length;
            start +=
                Math.max(
                    1,
                    sequenceLength -
                        32
                )
        ) {

            const part =
                tokens.slice(
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
            recursive:
                true
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

            updatedAt:
                new Date()
                    .toISOString(),

            model:
                config,

            training:
                {
                    epochs:
                        Number(
                            OPTIONS.epochs ||
                            10
                        ),

                    sequenceLength:
                        Number(
                            OPTIONS.sequenceLength ||
                            256
                        ),

                    learningRate:
                        Number(
                            OPTIONS.learningRate ||
                            0.00025
                        ),

                    contextSize:
                        Number(
                            OPTIONS.contextSize ||
                            256
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

            files:
                state.files,

            updatedAt:
                new Date()
                    .toISOString()
        }
    );
}


/* =========================================================
   TRAINING
========================================================= */

async function train(
    model,
    tokenizer,
    sequences,
    dataset,
    config
) {

    const epochs =
        Math.max(
            1,
            Number(
                OPTIONS.epochs ||
                10
            )
        );


    let step = 0;

    let loss = null;

    let bestLoss =
        Infinity;


    /*
      Vor dem eigentlichen neuronalen Training
      werden die kompletten Sequenzen in die
      Memory-Komponente des neuen Modells geladen.
    */

    send(
        "progress",
        {
            phase:
                "memory",

            progress:
                0,

            message:
                "Trainingswissen wird in die Modell-Memory geladen..."
        }
    );


    for (
        let i = 0;
        i <
            sequences.length;
        i++
    ) {

        if (stopRequested) {
            break;
        }


        const sequence =
            sequences[i];


        if (
            typeof model.rememberSequence ===
            "function"
        ) {

            model.rememberSequence(
                sequence
            );
        }


        if (
            i % 25 ===
            0
        ) {

            const progress =
                Math.round(
                    (
                        (
                            i + 1
                        ) /
                        sequences.length
                    ) *
                    100
                );


            send(
                "progress",
                {
                    phase:
                        "memory",

                    progress,

                    examples:
                        dataset.examples.length,

                    sequences:
                        sequences.length,

                    message:
                        "Trainingswissen wird übernommen..."
                }
            );


            await new Promise(
                resolve =>
                    setImmediate(
                        resolve
                    )
            );
        }
    }


    send(
        "progress",
        {
            phase:
                "memory",

            progress:
                100,

            message:
                "Trainingswissen übernommen."
        }
    );


    /*
      EIGENTLICHES TRAINING
    */

    for (
        let epoch = 1;
        epoch <= epochs;
        epoch++
    ) {

        if (stopRequested) {
            break;
        }


        let epochLoss =
            0;

        let epochTokens =
            0;


        for (
            let index = 0;
            index <
                sequences.length;
            index++
        ) {

            if (stopRequested) {
                break;
            }


            const sequence =
                sequences[index];


            if (
                !Array.isArray(
                    sequence
                ) ||
                sequence.length < 2
            ) {

                continue;
            }


            let result;


            try {

                /*
                  Wichtig:
                  Das komplette Sequence-Array wird übergeben.

                  Die neue model.js erzeugt daraus
                  selbst die nächsten Token.
                */

                result =
                    model.trainStep(
                        sequence
                    );


            } catch (error) {

                console.error(
                    "[TRAIN]",
                    error.message
                );


                send(
                    "warning",
                    {
                        message:
                            "Trainingsschritt fehlgeschlagen: " +
                            error.message
                    }
                );


                continue;
            }


            if (
                typeof result ===
                "number"
            ) {

                loss =
                    result;

            } else if (
                result &&
                typeof result.loss ===
                    "number"
            ) {

                loss =
                    result.loss;
            }


            if (
                Number.isFinite(
                    loss
                )
            ) {

                epochLoss +=
                    loss;

                epochTokens++;
            }


            step++;


            const progress =
                Math.round(
                    (
                        (
                            epoch - 1
                        ) /
                        epochs +
                        (
                            index + 1
                        ) /
                        sequences.length /
                        epochs
                    ) *
                    100
                );


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

                    examples:
                        dataset.examples.length,

                    sequences:
                        sequences.length,

                    message:
                        "Training: Epoche " +
                        epoch +
                        "/" +
                        epochs
                }
            );


            /*
              Event Loop freigeben,
              damit Render den Worker nicht
              als komplett blockiert behandelt.
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


        const averageLoss =
            epochTokens > 0
                ? epochLoss /
                    epochTokens
                : loss;


        loss =
            Number.isFinite(
                averageLoss
            )
                ? averageLoss
                : loss;


        if (
            Number.isFinite(
                loss
            ) &&
            loss <
                bestLoss
        ) {

            bestLoss =
                loss;
        }


        saveEverything(
            model,
            tokenizer,
            config,
            {
                epoch,

                step,

                loss,

                examples:
                    dataset.examples.length,

                sequences:
                    sequences.length,

                files:
                    dataset.files.length
            }
        );


        send(
            "epoch",
            {

                epoch,

                step,

                loss,

                bestLoss,

                totalEpochs:
                    epochs,

                examples:
                    dataset.examples.length,

                sequences:
                    sequences.length
            }
        );
    }


    return {

        epoch:
            stopRequested
                ? Math.max(
                    0,
                    Math.min(
                        epochs - 1,
                        epochs
                    )
                )
                : epochs,

        step,

        loss,

        bestLoss
    };
}


/* =========================================================
   START
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


        /*
          STATUS
        */

        send(
            "started",
            {

                phase:
                    "daten",

                progress:
                    0,

                message:
                    "Trainingsdaten werden aus DATEN geladen..."
            }
        );


        /*
          DATEN
        */

        const dataset =
            loadTrainingExamples();


        if (
            dataset.examples.length ===
            0
        ) {

            throw new Error(
                "Keine gültigen Trainingsdaten gefunden. " +
                "Lege mindestens eine JSON-Datei mit " +
                "frage/antwort bzw. question/answer in DATEN/ ab."
            );
        }


        send(
            "progress",
            {

                phase:
                    "daten",

                progress:
                    5,

                files:
                    dataset.files.length,

                validFiles:
                    dataset.validFiles.length,

                failedFiles:
                    dataset.failedFiles.length,

                examples:
                    dataset.examples.length,

                message:
                    dataset.examples.length +
                    " Trainingsbeispiele aus " +
                    dataset.files.length +
                    " JSON-Dateien gefunden."
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


        /*
          TOKENIZER
        */

        let tokenizer =
            loadTokenizer();


        tokenizer =
            await prepareTokenizer(
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


        /*
          SEQUENZEN
        */

        send(
            "progress",
            {

                phase:
                    "sequenzen",

                progress:
                    0,

                message:
                    "Trainingssequenzen werden erzeugt..."
            }
        );


        const sequenceLength =
            Math.max(
                16,
                Number(
                    OPTIONS.sequenceLength ||
                    256
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
                "Keine Trainingssequenzen konnten erstellt werden."
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


        /*
          MODELL
        */

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


        send(
            "progress",
            {

                phase:
                    "modell",

                progress:
                    0,

                message:
                    "LUMORA LARGE wird erstellt..."
            }
        );


        const {
            model,
            config
        } =
            createModel(
                tokenizer
            );


        const loadedExisting =
            loadExistingModel(
                model
            );


        send(
            "progress",
            {

                phase:
                    "modell",

                progress:
                    100,

                loadedExisting,

                parameters:
                    typeof model.parameterCount ===
                    "function"
                        ? model.parameterCount()
                        : null,

                config,

                message:
                    loadedExisting
                        ? "Vorhandenes kompatibles LUMORA-Modell geladen."
                        : "Neues LUMORA LARGE-Modell bereit."
            }
        );


        /*
          TRAINING
        */

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

                totalEpochs:
                    Number(
                        OPTIONS.epochs ||
                        10
                    ),

                examples:
                    dataset.examples.length,

                sequences:
                    sequences.length,

                message:
                    "Training gestartet."
            }
        );


        const result =
            await train(
                model,
                tokenizer,
                sequences,
                dataset,
                config
            );


        /*
          FINAL SPEICHERN
        */

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
                    sequences.length,

                files:
                    dataset.files.length
            }
        );


        /*
          FERTIG
        */

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

                    examples:
                        dataset.examples.length,

                    sequences:
                        sequences.length,

                    message:
                        "Training gestoppt. Der aktuelle Stand wurde gespeichert."
                }
            );

        } else {

            send(
                "finished",
                {

                    epoch:
                        result.epoch,

                    step:
                        result.step,

                    loss:
                        result.loss,

                    bestLoss:
                        result.bestLoss,

                    progress:
                        100,

                    examples:
                        dataset.examples.length,

                    sequences:
                        sequences.length,

                    files:
                        dataset.files.length,

                    message:
                        "LUMORA LARGE wurde erfolgreich trainiert und gespeichert."
                }
            );
        }

    } catch (error) {

        console.error(
            "======================================"
        );

        console.error(
            "LUMORA TRAINING FEHLER"
        );

        console.error(
            error
        );

        console.error(
            "======================================"
        );


        send(
            "error",
            {

                error:
                    error.message ||
                    String(error),

                stack:
                    error.stack ||

                    null
            }
        );
    }
}


main();

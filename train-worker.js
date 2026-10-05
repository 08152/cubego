"use strict";

const fs = require("fs");
const path = require("path");
const { parentPort, workerData } = require("worker_threads");

const ROOT = workerData.root || __dirname;
const DATEN = workerData.daten || path.join(ROOT, "DATEN");
const GELERNT = workerData.gelernt || path.join(ROOT, "GELERNT");
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

function send(type, data = {}) {
    if (parentPort) {
        parentPort.postMessage({
            type,
            ...data
        });
    }
}

function writeJSON(file, data) {
    fs.mkdirSync(path.dirname(file), {
        recursive: true
    });

    const temp = file + ".tmp";

    fs.writeFileSync(
        temp,
        JSON.stringify(data, null, 2),
        "utf8"
    );

    fs.renameSync(temp, file);
}

function readJSON(file) {
    return JSON.parse(
        fs.readFileSync(file, "utf8")
    );
}


/* =========================================================
   DATEN LADEN
   ========================================================= */

function getDataFiles() {
    if (!fs.existsSync(DATEN)) {
        fs.mkdirSync(DATEN, {
            recursive: true
        });

        return [];
    }

    return fs.readdirSync(DATEN)
        .filter(name => {
            const lower = name.toLowerCase();

            return (
                lower.endsWith(".json") ||
                lower.endsWith(".jsonl") ||
                lower.endsWith(".txt")
            );
        });
}

function loadTrainingExamples() {
    const files = getDataFiles();
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

    function scan(value) {
        if (stopRequested) return;

        if (Array.isArray(value)) {
            for (const item of value) {
                scan(item);
            }

            return;
        }

        if (!value || typeof value !== "object") {
            return;
        }

        let question = null;
        let answer = null;

        for (const key of Object.keys(value)) {
            const lower = key.toLowerCase();

            if (questionKeys.includes(lower)) {
                question = value[key];
            }

            if (answerKeys.includes(lower)) {
                answer = value[key];
            }
        }

        if (
            typeof question === "string" &&
            typeof answer === "string" &&
            question.trim() &&
            answer.trim()
        ) {
            examples.push({
                question: question.trim(),
                answer: answer.trim()
            });
        }

        for (const key of Object.keys(value)) {
            scan(value[key]);
        }
    }

    for (const filename of files) {
        if (stopRequested) break;

        const file = path.join(
            DATEN,
            filename
        );

        let raw;

        try {
            raw = fs.readFileSync(
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

        if (
            filename.toLowerCase().endsWith(".txt")
        ) {
            if (raw.trim()) {
                examples.push({
                    question:
                        "Erzähle etwas über diesen Text.",
                    answer:
                        raw.trim()
                });
            }

            continue;
        }

        try {
            if (
                filename.toLowerCase().endsWith(".jsonl")
            ) {
                const lines = raw
                    .split(/\r?\n/)
                    .filter(Boolean);

                for (const line of lines) {
                    try {
                        scan(JSON.parse(line));
                    } catch {}
                }
            } else {
                scan(JSON.parse(raw));
            }
        } catch (error) {
            console.error(
                "JSON konnte nicht verarbeitet werden:",
                filename,
                error.message
            );
        }
    }

    const unique = [];
    const seen = new Set();

    for (const example of examples) {
        const id =
            example.question +
            "\n" +
            example.answer;

        if (!seen.has(id)) {
            seen.add(id);
            unique.push(example);
        }
    }

    return {
        files,
        examples: unique
    };
}


/* =========================================================
   TOKENIZER LADEN
   ========================================================= */

function loadTokenizer() {
    const tokenizerFile =
        path.join(
            ROOT,
            "tokenizer.js"
        );

    if (!fs.existsSync(tokenizerFile)) {
        throw new Error(
            "tokenizer.js wurde nicht gefunden."
        );
    }

    delete require.cache[
        require.resolve(tokenizerFile)
    ];

    const loaded =
        require(tokenizerFile);

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

/*
   WICHTIG:

   Wir rufen hier NICHT mehr blind tokenizer.trainBPE()
   auf.

   Genau dieser Aufruf hat bei Render beim kleinen Datensatz
   den Worker blockiert.

   Stattdessen wird versucht, vorhandene Trainingsfunktionen
   kontrolliert zu verwenden. Falls keine sichere Funktion
   verfügbar ist, benutzen wir den vorhandenen Tokenizer direkt.
*/

async function trainTokenizerSafe(
    tokenizer,
    examples
) {
    send(
        "progress",
        {
            phase: "tokenizer",
            progress: 10,
            message:
                "Tokenizer wird vorbereitet..."
        }
    );

    if (stopRequested) {
        return tokenizer;
    }

    const texts = examples.map(example =>
        "<|user|>\n" +
        example.question +
        "\n<|end|>\n" +
        "<|assistant|>\n" +
        example.answer +
        "\n<|end|>"
    );

    /*
       Bei kleinen Datensätzen ist ein vorhandener Tokenizer
       ausreichend für den ersten Test.
    */

    if (
        typeof tokenizer.trainBPE ===
        "function"
    ) {
        send(
            "progress",
            {
                phase: "tokenizer",
                progress: 20,
                message:
                    "BPE-Tokenizer wird mit kleinem Datensatz vorbereitet..."
            }
        );

        /*
           Nur bei ausreichend Daten BPE starten.
           Mit den aktuellen 20 Beispielen wird es übersprungen,
           damit Render nicht hängen bleibt.
        */

        if (
            texts.length >= 100
        ) {
            try {
                await Promise.race([
                    Promise.resolve(
                        tokenizer.trainBPE(
                            texts
                        )
                    ),
                    new Promise(
                        (_, reject) =>
                            setTimeout(
                                () =>
                                    reject(
                                        new Error(
                                            "BPE-Training Timeout"
                                        )
                                    ),
                                30000
                            )
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
                            "BPE übersprungen – vorhandener Tokenizer wird verwendet."
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
                        "Kleiner Datensatz erkannt – BPE wird übersprungen."
                }
            );
        }
    } else if (
        typeof tokenizer.train ===
        "function" &&
        texts.length >= 100
    ) {
        try {
            await Promise.race([
                Promise.resolve(
                    tokenizer.train(
                        texts
                    )
                ),
                new Promise(
                    (_, reject) =>
                        setTimeout(
                            () =>
                                reject(
                                    new Error(
                                        "Tokenizer-Training Timeout"
                                    )
                                ),
                            30000
                        )
                )
            ]);
        } catch (error) {
            console.warn(
                "Tokenizer-Training übersprungen:",
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

function exportTokenizer(
    tokenizer
) {
    let data;

    if (
        typeof tokenizer.export ===
        "function"
    ) {
        data = tokenizer.export();
    } else if (
        typeof tokenizer.toJSON ===
        "function"
    ) {
        data = tokenizer.toJSON();
    } else {
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
   MODELL
   ========================================================= */

function loadModelClass() {
    const modelFile =
        path.join(
            ROOT,
            "model.js"
        );

    if (!fs.existsSync(modelFile)) {
        throw new Error(
            "model.js wurde nicht gefunden."
        );
    }

    delete require.cache[
        require.resolve(modelFile)
    ];

    const loaded =
        require(modelFile);

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

    const vocabSize =
        Number(
            tokenizer.vocabSize ||
            tokenizer.vocab?.length ||
            8192
        );

    const config = {
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

        dropout:
            0,

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

    const model =
        new Model(config);

    return {
        model,
        config
    };
}


/* =========================================================
   TRAININGSSEQUENZEN
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

        if (!tokens) {
            continue;
        }

        const list =
            Array.from(tokens);

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
                1,

            updatedAt:
                new Date().toISOString(),

            model:
                config,

            training: {
                epochs:
                    OPTIONS.epochs ||
                    10,

                sequenceLength:
                    OPTIONS.sequenceLength ||
                    256,

                learningRate:
                    OPTIONS.learningRate ||
                    0.0003
            }
        }
    );

    writeJSON(
        STATE_FILE,
        {
            version:
                1,

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
    tokenizer,
    sequences,
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
                                learningRate:
                                    config.learningRate
                            }
                        );
                } else if (
                    typeof model.trainBackprop ===
                    "function"
                ) {
                    result =
                        model.trainBackprop(
                            input,
                            target,
                            {
                                learningRate:
                                    config.learningRate
                            }
                        );
                } else {
                    throw new Error(
                        "trainStep/trainBackprop fehlt."
                    );
                }

                if (
                    typeof result ===
                    "number"
                ) {
                    loss = result;
                } else if (
                    result &&
                    typeof result.loss ===
                    "number"
                ) {
                    loss = result.loss;
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
                            epoch -
                            1
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

                    message:
                        "Training: Epoche " +
                        epoch +
                        "/" +
                        epochs
                }
            );

            if (
                step % 10 ===
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

        saveEverything(
            model,
            tokenizer,
            config,
            {
                epoch,
                step,
                loss,
                examples:
                    sequences.length,
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
            Math.min(
                epochs,
                epochSafe(
                    epochs,
                    stopRequested
                )
            ),

        step,

        loss
    };
}

function epochSafe(
    epochs,
    stopped
) {
    if (stopped) {
        return Math.max(
            0,
            epochs - 1
        );
    }

    return epochs;
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

                message:
                    dataset.examples.length +
                    " Trainingsbeispiele gefunden."
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

        /* TOKENIZER */

        const tokenizer =
            loadTokenizer();

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

        /* SEQUENZEN */

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
            Number(
                OPTIONS.sequenceLength ||
                256
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

        /* MODELL */

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
                    "Modell wird erstellt..."
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

                message:
                    "Modell bereit."
            }
        );

        /* TRAINING */

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
                    "Modelltraining gestartet..."
            }
        );

        const result =
            await train(
                model,
                tokenizer,
                sequences,
                config
            );

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
                        "Training gestoppt. Daten wurden gespeichert."
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

                    progress:
                        100,

                    message:
                        "Training vollständig abgeschlossen."
                }
            );
        }

    } catch (error) {
        console.error(
            "TRAIN-WORKER ERROR:",
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

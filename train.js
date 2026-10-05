/* ============================================================
   train.js
   ============================================================
   TRAININGSSYSTEM FÜR DAS EIGENE SPRACHMODELL

   Liest automatisch:
       DATEN/*.json

   Unterstützt z.B.:
       DATEN/training.json
       DATEN/wissen.json
       DATEN/sprechen.json
       DATEN/regeln.json
       DATEN/persöhnlichkeit.json
       DATEN/weitere_datei.json

   Keine externen Bibliotheken.
   ============================================================ */

"use strict";


/* ============================================================
   KONFIGURATION
   ============================================================ */

const TRAIN_CONFIG = {

    dataFolder: "DATEN/",

    /*
     * Dateien, die automatisch versucht werden.
     * Zusätzliche JSON-Dateien können über den
     * Ordner-Auswahldialog geladen werden.
     */

    defaultFiles: [
        "training.json",
        "wissen.json",
        "wissen_de.json",
        "sprechen.json",
        "regeln.json",
        "persöhnlichkeit.json",
        "persoenlichkeit.json",
        "dialoge.json",
        "fragen.json",
        "antworten.json",
        "daten.json",
        "chat.json"
    ],

    epochs: 10,

    batchSize: 1,

    learningRate: 0.0003,

    sequenceLength: 256,

    shuffle: true,

    saveEveryEpoch: true,

    modelStorageKey: "MEINE_KI_MODEL",

    tokenizerStorageKey: "MEINE_KI_TOKENIZER",

    logEvery: 1
};


/* ============================================================
   STATUS
   ============================================================ */

const TRAIN_STATUS = {

    running: false,

    epoch: 0,

    totalEpochs: 0,

    currentFile: "",

    currentExample: 0,

    totalExamples: 0,

    loss: 0,

    averageLoss: 0,

    tokens: 0,

    files: [],

    examples: 0,

    startedAt: 0,

    elapsed: 0,

    stopped: false
};


/* ============================================================
   EVENT SYSTEM
   ============================================================ */

const TrainEvents = {

    listeners: {},

    on(name, callback) {

        if (!this.listeners[name]) {
            this.listeners[name] = [];
        }

        this.listeners[name].push(callback);
    },

    emit(name, data) {

        const list =
            this.listeners[name] || [];

        for (const callback of list) {

            try {
                callback(data);
            } catch (error) {
                console.error(
                    "TrainEvents:",
                    error
                );
            }
        }
    }
};


/* ============================================================
   LOG
   ============================================================ */

function trainLog(...args) {

    console.log(
        "[TRAIN]",
        ...args
    );

    TrainEvents.emit(
        "log",
        args.join(" ")
    );
}


/* ============================================================
   JSON HERUNTERLADEN
   ============================================================ */

async function loadJSONFile(
    filename
) {

    const path =
        TRAIN_CONFIG.dataFolder +
        filename;


    try {

        const response =
            await fetch(path, {
                cache: "no-store"
            });


        if (!response.ok) {

            return null;
        }


        const text =
            await response.text();


        if (!text.trim()) {

            return null;
        }


        return JSON.parse(
            text
        );

    } catch (error) {

        console.warn(
            "JSON konnte nicht geladen werden:",
            path,
            error
        );

        return null;
    }
}


/* ============================================================
   ALLE STANDARD-DATEIEN LADEN
   ============================================================ */

async function loadDefaultDataFiles() {

    const result = [];

    for (
        const filename of
        TRAIN_CONFIG.defaultFiles
    ) {

        const data =
            await loadJSONFile(
                filename
            );


        if (
            data === null
        ) {
            continue;
        }


        result.push({

            name: filename,

            data
        });


        trainLog(
            "Geladen:",
            filename
        );
    }


    return result;
}


/* ============================================================
   ORDNER AUS DATEIAUSWAHL LADEN
   ============================================================ */

async function loadDataFolderFromPicker() {

    return new Promise(
        (resolve) => {

            const input =
                document.createElement(
                    "input"
                );


            input.type = "file";

            input.multiple = true;

            input.accept = ".json";


            /*
             * Chrome unterstützt:
             * webkitdirectory
             */

            input.webkitdirectory =
                true;


            input.onchange =
                async function () {

                    const files =
                        Array.from(
                            input.files || []
                        );


                    const jsonFiles =
                        files.filter(
                            file =>
                                file.name
                                    .toLowerCase()
                                    .endsWith(".json")
                        );


                    const result = [];


                    for (
                        const file of
                        jsonFiles
                    ) {

                        try {

                            const text =
                                await file.text();


                            const data =
                                JSON.parse(
                                    text
                                );


                            result.push({

                                name:
                                    file.webkitRelativePath ||
                                    file.name,

                                data
                            });


                        } catch (error) {

                            console.warn(
                                "Fehler in:",
                                file.name,
                                error
                            );
                        }
                    }


                    resolve(
                        result
                    );
                };


            input.click();
        }
    );
}


/* ============================================================
   REKURSIV TEXT AUS BELIEBIGEM JSON HOLEN
   ============================================================ */

function extractTextFromJSON(
    value,
    output,
    path
) {

    if (
        value === null ||
        value === undefined
    ) {
        return;
    }


    /*
     * String
     */

    if (
        typeof value ===
        "string"
    ) {

        const text =
            value.trim();


        if (
            text.length > 0
        ) {

            output.push({

                text,

                path:
                    path || ""
            });
        }


        return;
    }


    /*
     * Zahl
     */

    if (
        typeof value ===
        "number"
    ) {

        output.push({

            text:
                String(value),

            path:
                path || ""
        });


        return;
    }


    /*
     * Boolean
     */

    if (
        typeof value ===
        "boolean"
    ) {

        output.push({

            text:
                value
                    ? "wahr"
                    : "falsch",

            path:
                path || ""
        });


        return;
    }


    /*
     * Array
     */

    if (
        Array.isArray(value)
    ) {

        for (
            let i = 0;
            i < value.length;
            i++
        ) {

            extractTextFromJSON(
                value[i],
                output,
                `${path}[${i}]`
            );
        }


        return;
    }


    /*
     * Objekt
     */

    if (
        typeof value ===
        "object"
    ) {

        for (
            const key of
            Object.keys(value)
        ) {

            const child =
                value[key];


            /*
             * Bei typischen Chatdaten
             * versuchen wir die Struktur
             * sinnvoll zu erhalten.
             */

            if (
                typeof child ===
                "string"
            ) {

                output.push({

                    text:
                        `${key}: ${child}`,

                    path:
                        path
                            ? `${path}.${key}`
                            : key
                });

            } else {

                extractTextFromJSON(
                    child,
                    output,
                    path
                        ? `${path}.${key}`
                        : key
                );
            }
        }
    }
}


/* ============================================================
   JSON → TEXTDATEN
   ============================================================ */

function convertFilesToText(
    files
) {

    const examples = [];


    for (
        const file of
        files
    ) {

        const extracted = [];


        extractTextFromJSON(
            file.data,
            extracted,
            ""
        );


        /*
         * Datei-Kontext hinzufügen
         */

        for (
            const item of
            extracted
        ) {

            const text =
                `[DATEI: ${file.name}]\n` +
                item.text;


            examples.push({

                text,

                file:
                    file.name,

                path:
                    item.path
            });
        }
    }


    return examples;
}


/* ============================================================
   SPEZIELLE TRAININGSPAARE ERKENNEN
   ============================================================ */

function createConversationText(
    value
) {

    if (
        !value ||
        typeof value !== "object"
    ) {

        return null;
    }


    /*
     * Häufige Form:
     *
     * {
     *   "frage": "...",
     *   "antwort": "..."
     * }
     */

    const question =
        value.frage ??
        value.question ??
        value.user ??
        value.input;


    const answer =
        value.antwort ??
        value.answer ??
        value.assistant ??
        value.output;


    if (
        typeof question === "string" &&
        typeof answer === "string"
    ) {

        return (
            "<|user|>\n" +
            question.trim() +
            "\n<|end|>\n" +

            "<|assistant|>\n" +
            answer.trim() +
            "\n<|end|>"
        );
    }


    return null;
}


/* ============================================================
   KONVERSATIONSSTRUKTUREN ERKENNEN
   ============================================================ */

function extractConversationPairs(
    value,
    output
) {

    if (
        !value ||
        typeof value !== "object"
    ) {
        return;
    }


    if (
        Array.isArray(value)
    ) {

        for (
            const item of
            value
        ) {

            const conversation =
                createConversationText(
                    item
                );


            if (
                conversation
            ) {

                output.push(
                    conversation
                );
            }


            extractConversationPairs(
                item,
                output
            );
        }


        return;
    }


    const conversation =
        createConversationText(
            value
        );


    if (
        conversation
    ) {

        output.push(
            conversation
        );
    }


    for (
        const key of
        Object.keys(value)
    ) {

        extractConversationPairs(
            value[key],
            output
        );
    }
}


/* ============================================================
   TEXT NORMALISIEREN
   ============================================================ */

function normalizeTrainingText(
    text
) {

    return String(text || "")
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n")
        .replace(/[ \t]+/g, " ")
        .replace(/\n{4,}/g, "\n\n")
        .trim();
}


/* ============================================================
   TRAININGSDATEN ERSTELLEN
   ============================================================ */

function buildTrainingDataset(
    files
) {

    const examples = [];

    const conversations = [];


    /*
     * Zuerst echte Frage/Antwort-Paare
     */

    for (
        const file of
        files
    ) {

        extractConversationPairs(
            file.data,
            conversations
        );
    }


    for (
        const conversation of
        conversations
    ) {

        const text =
            normalizeTrainingText(
                conversation
            );


        if (
            text.length > 0
        ) {

            examples.push({

                text,

                type:
                    "conversation",

                file:
                    "conversation"
            });
        }
    }


    /*
     * Danach alle übrigen Textdaten
     */

    const generic =
        convertFilesToText(
            files
        );


    for (
        const item of
        generic
    ) {

        const text =
            normalizeTrainingText(
                item.text
            );


        if (
            text.length > 0
        ) {

            examples.push({

                text,

                type:
                    "knowledge",

                file:
                    item.file,

                path:
                    item.path
            });
        }
    }


    /*
     * Duplikate entfernen
     */

    const unique =
        new Map();


    for (
        const item of
        examples
    ) {

        const key =
            item.text;


        if (
            !unique.has(key)
        ) {

            unique.set(
                key,
                item
            );
        }
    }


    return Array.from(
        unique.values()
    );
}


/* ============================================================
   TRAININGSTEXTE IN SEQUENZEN AUFTEILEN
   ============================================================ */

function createSequences(
    tokenizer,
    examples,
    sequenceLength
) {

    const sequences = [];


    for (
        const example of
        examples
    ) {

        let tokens;


        try {

            tokens =
                tokenizer.encode(
                    example.text
                );

        } catch (error) {

            console.warn(
                "Tokenisierung fehlgeschlagen:",
                example.text,
                error
            );

            continue;
        }


        if (
            !tokens ||
            tokens.length < 2
        ) {
            continue;
        }


        /*
         * Lange Texte in mehrere
         * Trainingsabschnitte zerlegen.
         */

        for (
            let start = 0;
            start < tokens.length - 1;
            start += sequenceLength - 1
        ) {

            const part =
                tokens.slice(
                    start,
                    start + sequenceLength
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


/* ============================================================
   SHUFFLE
   ============================================================ */

function shuffleArray(
    array
) {

    for (
        let i = array.length - 1;
        i > 0;
        i--
    ) {

        const j =
            Math.floor(
                Math.random() *
                (i + 1)
            );


        [
            array[i],
            array[j]
        ] =
        [
            array[j],
            array[i]
        ];
    }


    return array;
}


/* ============================================================
   TOKENIZER PRÜFEN
   ============================================================ */

function getTokenizer() {

    if (
        typeof window !==
        "undefined" &&
        window.AdvancedTokenizer
    ) {

        return window.AdvancedTokenizer;
    }


    if (
        typeof AdvancedTokenizer !==
        "undefined"
    ) {

        return AdvancedTokenizer;
    }


    throw new Error(
        "AdvancedTokenizer wurde nicht gefunden. " +
        "tokenizer.js muss vor train.js geladen werden."
    );
}


/* ============================================================
   TOKENIZER INSTANZ ERSTELLEN
   ============================================================ */

async function createTokenizer() {

    const Tokenizer =
        getTokenizer();


    /*
     * Unterstützt:
     *
     * AdvancedTokenizer.create()
     * new AdvancedTokenizer()
     */

    if (
        typeof Tokenizer.create ===
        "function"
    ) {

        return await Tokenizer.create();
    }


    return new Tokenizer();
}


/* ============================================================
   MODELL PRÜFEN
   ============================================================ */

function getModelClass() {

    if (
        typeof window !==
        "undefined" &&
        window.LanguageModel
    ) {

        return window.LanguageModel;
    }


    if (
        typeof LanguageModel !==
        "undefined"
    ) {

        return LanguageModel;
    }


    throw new Error(
        "LanguageModel wurde nicht gefunden. " +
        "model.js muss vor train.js geladen werden."
    );
}


/* ============================================================
   MODELL ERSTELLEN
   ============================================================ */

function createTrainingModel(
    tokenizer
) {

    const Model =
        getModelClass();


    const vocabSize =
        tokenizer.vocabSize ||
        tokenizer.config?.vocabSize ||
        TRAIN_CONFIG.vocabSize ||
        8192;


    return new Model({

        vocabSize,

        contextSize:
            TRAIN_CONFIG.sequenceLength,

        embeddingSize:
            192,

        layers:
            6,

        heads:
            6,

        headSize:
            32,

        feedForwardSize:
            512,

        learningRate:
            TRAIN_CONFIG.learningRate,

        gradientClip:
            1.0,

        temperature:
            0.85,

        topK:
            40,

        topP:
            0.92
    });
}


/* ============================================================
   TOKENIZER SPEICHERN
   ============================================================ */

function saveTokenizer(
    tokenizer
) {

    if (
        typeof localStorage ===
        "undefined"
    ) {
        return;
    }


    try {

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

            data =
                JSON.stringify(
                    tokenizer
                );
        }


        localStorage.setItem(
            TRAIN_CONFIG.tokenizerStorageKey,
            typeof data === "string"
                ? data
                : JSON.stringify(data)
        );

    } catch (error) {

        console.warn(
            "Tokenizer konnte nicht gespeichert werden:",
            error
        );
    }
}


/* ============================================================
   MODELL SPEICHERN
   ============================================================ */

function saveModel(
    model
) {

    if (
        typeof localStorage ===
        "undefined"
    ) {
        return;
    }


    try {

        model.saveLocalStorage(
            TRAIN_CONFIG.modelStorageKey
        );

        trainLog(
            "Modell gespeichert."
        );

    } catch (error) {

        console.error(
            "Modell speichern:",
            error
        );
    }
}


/* ============================================================
   TRAININGSFORTSCHRITT
   ============================================================ */

function emitProgress(
    extra
) {

    const data =
        Object.assign(
            {},
            TRAIN_STATUS,
            extra || {}
        );


    TrainEvents.emit(
        "progress",
        data
    );
}


/* ============================================================
   TRAINING STOPPEN
   ============================================================ */

function stopTraining() {

    TRAIN_STATUS.stopped =
        true;

    TRAIN_STATUS.running =
        false;


    TrainEvents.emit(
        "stopped",
        TRAIN_STATUS
    );


    trainLog(
        "Training wird beendet..."
    );
}


/* ============================================================
   EPOCH TRAINIEREN
   ============================================================ */

async function trainEpoch(
    model,
    sequences,
    epoch
) {

    let totalLoss = 0;

    let count = 0;

    let totalGradient = 0;


    for (
        let i = 0;
        i < sequences.length;
        i++
    ) {

        if (
            TRAIN_STATUS.stopped
        ) {
            break;
        }


        const tokens =
            sequences[i];


        try {

            const result =
                model.trainStep(
                    tokens
                );


            const loss =
                Number(
                    result.loss || 0
                );


            totalLoss +=
                loss;


            totalGradient +=
                Number(
                    result.gradientNorm || 0
                );


            count++;


            TRAIN_STATUS.currentExample =
                i + 1;


            TRAIN_STATUS.loss =
                loss;


            TRAIN_STATUS.averageLoss =
                totalLoss /
                Math.max(
                    count,
                    1
                );


            emitProgress({

                epoch,

                currentExample:
                    i + 1,

                totalExamples:
                    sequences.length,

                loss,

                averageLoss:
                    TRAIN_STATUS.averageLoss,

                gradientNorm:
                    result.gradientNorm || 0
            });


            /*
             * Browser nicht komplett einfrieren.
             */

            if (
                i % 2 === 0
            ) {

                await new Promise(
                    resolve =>
                        setTimeout(
                            resolve,
                            0
                        )
                );
            }

        } catch (error) {

            console.error(
                "Training-Fehler:",
                error
            );


            TrainEvents.emit(
                "error",
                error
            );
        }
    }


    return {

        loss:
            count
                ? totalLoss / count
                : 0,

        gradientNorm:
            count
                ? totalGradient / count
                : 0
    };
}


/* ============================================================
   KOMPLETTES TRAINING
   ============================================================ */

async function trainAI(
    options
) {

    options =
        Object.assign(
            {},
            TRAIN_CONFIG,
            options || {}
        );


    if (
        TRAIN_STATUS.running
    ) {

        throw new Error(
            "Training läuft bereits."
        );
    }


    TRAIN_STATUS.running =
        true;

    TRAIN_STATUS.stopped =
        false;

    TRAIN_STATUS.epoch =
        0;

    TRAIN_STATUS.startedAt =
        Date.now();


    trainLog(
        "========================================"
    );

    trainLog(
        "EIGENES KI-TRAINING START"
    );

    trainLog(
        "========================================"
    );


    /*
     * Tokenizer
     */

    trainLog(
        "Tokenizer wird vorbereitet..."
    );


    const tokenizer =
        await createTokenizer();


    /*
     * Daten laden
     */

    trainLog(
        "Lade alle DATEN/*.json ..."
    );


    let files =
        await loadDefaultDataFiles();


    /*
     * Falls automatisch keine Dateien
     * gefunden wurden, Dateiauswahl öffnen.
     */

    if (
        files.length === 0
    ) {

        trainLog(
            "Keine Standarddateien gefunden."
        );


        if (
            typeof document !==
            "undefined"
        ) {

            trainLog(
                "Bitte DATEN-Ordner auswählen."
            );


            files =
                await loadDataFolderFromPicker();
        }
    }


    if (
        files.length === 0
    ) {

        TRAIN_STATUS.running =
            false;


        throw new Error(
            "Keine JSON-Trainingsdaten gefunden."
        );
    }


    TRAIN_STATUS.files =
        files.map(
            file =>
                file.name
        );


    /*
     * Datensatz
     */

    trainLog(
        "Erstelle Trainingsdatensatz..."
    );


    const examples =
        buildTrainingDataset(
            files
        );


    TRAIN_STATUS.examples =
        examples.length;


    trainLog(
        "Beispiele:",
        examples.length
    );


    /*
     * Sequenzen
     */

    trainLog(
        "Tokenisiere Trainingsdaten..."
    );


    const sequences =
        createSequences(
            tokenizer,
            examples,
            options.sequenceLength
        );


    TRAIN_STATUS.tokens =
        sequences.reduce(
            (
                total,
                sequence
            ) =>
                total +
                sequence.length,
            0
        );


    TRAIN_STATUS.totalExamples =
        sequences.length;


    trainLog(
        "Trainingssequenzen:",
        sequences.length
    );


    trainLog(
        "Tokens:",
        TRAIN_STATUS.tokens
    );


    /*
     * Modell
     */

    trainLog(
        "Erstelle Transformer..."
    );


    const model =
        createTrainingModel(
            tokenizer
        );


    trainLog(
        "Parameter:",
        model.parameterCount()
    );


    trainLog(
        "Parameter in Millionen:",
        (
            model.parameterCount() /
            1000000
        ).toFixed(2)
    );


    /*
     * Training
     */

    for (
        let epoch = 1;
        epoch <= options.epochs;
        epoch++
    ) {

        if (
            TRAIN_STATUS.stopped
        ) {
            break;
        }


        TRAIN_STATUS.epoch =
            epoch;


        trainLog(
            `Epoch ${epoch}/${options.epochs}`
        );


        if (
            options.shuffle
        ) {

            shuffleArray(
                sequences
            );
        }


        const result =
            await trainEpoch(
                model,
                sequences,
                epoch
            );


        trainLog(
            "Loss:",
            result.loss.toFixed(6)
        );


        trainLog(
            "Gradient:",
            result.gradientNorm.toFixed(6)
        );


        if (
            options.saveEveryEpoch
        ) {

            saveModel(
                model
            );

            saveTokenizer(
                tokenizer
            );
        }


        /*
         * GC/Browser Zeit geben
         */

        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    10
                )
        );
    }


    TRAIN_STATUS.running =
        false;

    TRAIN_STATUS.elapsed =
        Date.now() -
        TRAIN_STATUS.startedAt;


    /*
     * Final speichern
     */

    saveModel(
        model
    );

    saveTokenizer(
        tokenizer
    );


    trainLog(
        "========================================"
    );

    trainLog(
        "TRAINING BEENDET"
    );

    trainLog(
        "========================================"
    );


    TrainEvents.emit(
        "complete",
        {
            model,
            tokenizer,
            status:
                Object.assign(
                    {},
                    TRAIN_STATUS
                )
        }
    );


    return {

        model,

        tokenizer,

        examples,

        sequences,

        status:
            Object.assign(
                {},
                TRAIN_STATUS
            )
    };
}


/* ============================================================
   TRAINING AUS GESPEICHERTEM MODELL FORTSETZEN
   ============================================================ */

function loadSavedModel() {

    const Model =
        getModelClass();


    const tokenizerClass =
        getTokenizer();


    let tokenizer =
        null;


    /*
     * Tokenizer laden
     */

    if (
        typeof localStorage !==
        "undefined"
    ) {

        const tokenizerData =
            localStorage.getItem(
                TRAIN_CONFIG.tokenizerStorageKey
            );


        if (
            tokenizerData
        ) {

            try {

                if (
                    typeof tokenizerClass
                        .fromJSON ===
                    "function"
                ) {

                    tokenizer =
                        tokenizerClass.fromJSON(
                            tokenizerData
                        );

                } else {

                    tokenizer =
                        new tokenizerClass();
                }

            } catch (error) {

                console.warn(
                    "Tokenizer laden fehlgeschlagen:",
                    error
                );
            }
        }
    }


    /*
     * Modell
     */

    let model =
        new Model();


    try {

        if (
            typeof localStorage !==
            "undefined"
        ) {

            const modelData =
                localStorage.getItem(
                    TRAIN_CONFIG.modelStorageKey
                );


            if (
                modelData
            ) {

                model.load(
                    JSON.parse(
                        modelData
                    )
                );
            }
        }

    } catch (error) {

        console.warn(
            "Modell laden fehlgeschlagen:",
            error
        );
    }


    return {

        model,

        tokenizer
    };
}


/* ============================================================
   TRAININGSVORSCHAU
   ============================================================ */

async function previewTrainingData() {

    const files =
        await loadDefaultDataFiles();


    const examples =
        buildTrainingDataset(
            files
        );


    return {

        files:
            files.map(
                x => x.name
            ),

        examples:
            examples.length,

        preview:
            examples
                .slice(0, 20)
                .map(
                    x => x.text
                )
    };
}


/* ============================================================
   TEST NACH DEM TRAINING
   ============================================================ */

function testModel(
    model,
    tokenizer,
    prompt
) {

    if (
        !model ||
        !tokenizer
    ) {

        throw new Error(
            "Model und Tokenizer erforderlich."
        );
    }


    const result =
        model.generate(
            prompt,
            tokenizer,
            {
                maxTokens: 100,

                temperature:
                    0.8,

                topK:
                    40,

                topP:
                    0.92,

                repetitionPenalty:
                    1.08
            }
        );


    trainLog(
        "PROMPT:",
        prompt
    );


    trainLog(
        "ANTWORT:",
        result
    );


    return result;
}


/* ============================================================
   GLOBAL
   ============================================================ */

if (
    typeof window !==
    "undefined"
) {

    window.TRAIN_CONFIG =
        TRAIN_CONFIG;

    window.TRAIN_STATUS =
        TRAIN_STATUS;

    window.TrainEvents =
        TrainEvents;

    window.trainAI =
        trainAI;

    window.stopTraining =
        stopTraining;

    window.loadDefaultDataFiles =
        loadDefaultDataFiles;

    window.loadDataFolderFromPicker =
        loadDataFolderFromPicker;

    window.buildTrainingDataset =
        buildTrainingDataset;

    window.createSequences =
        createSequences;

    window.previewTrainingData =
        previewTrainingData;

    window.loadSavedModel =
        loadSavedModel;

    window.testModel =
        testModel;
}


/* ============================================================
   NODE.JS
   ============================================================ */

if (
    typeof module !== "undefined" &&
    module.exports
) {

    module.exports = {

        TRAIN_CONFIG,

        TRAIN_STATUS,

        TrainEvents,

        trainAI,

        stopTraining,

        loadDefaultDataFiles,

        loadDataFolderFromPicker,

        buildTrainingDataset,

        createSequences,

        previewTrainingData,

        loadSavedModel,

        testModel
    };
}

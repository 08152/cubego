"use strict";

/*
================================================================
  generate-worker.js
  LUMORA
  - lädt GELERNT/model.json
  - lädt GELERNT/tokenizer.json
  - erzeugt echte Textantworten
  - sendet beim Stream TEXT statt Token-IDs
  - erkennt Spezialtokens zuverlässig
  - kompatibel mit model.js + tokenizer.js
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

const GELERNT =
    workerData.gelernt ||
    path.join(
        ROOT,
        "GELERNT"
    );

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


/* =========================================================
   STOP
   ========================================================= */

let stopped = false;

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
                stopped = true;
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

function readJSON(file) {
    return JSON.parse(
        fs.readFileSync(
            file,
            "utf8"
        )
    );
}


/* =========================================================
   DATEI PRÜFEN
   ========================================================= */

function requireFile(
    file,
    message
) {
    if (
        !fs.existsSync(file)
    ) {
        throw new Error(
            message
        );
    }
}


/* =========================================================
   TOKENIZER
   ========================================================= */

function loadTokenizerModule() {

    const file =
        path.join(
            ROOT,
            "tokenizer.js"
        );

    requireFile(
        file,
        "tokenizer.js wurde nicht gefunden."
    );

    delete require.cache[
        require.resolve(file)
    ];

    const loaded =
        require(file);

    if (!loaded) {
        throw new Error(
            "tokenizer.js hat keinen Export."
        );
    }

    return loaded;
}


function createTokenizer() {

    const loaded =
        loadTokenizerModule();

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


function loadTokenizer() {

    requireFile(
        TOKENIZER_FILE,
        "GELERNT/tokenizer.json wurde nicht gefunden. " +
        "Trainiere LUMORA zuerst."
    );

    const tokenizer =
        createTokenizer();

    const saved =
        readJSON(
            TOKENIZER_FILE
        );

    if (
        typeof tokenizer.import ===
            "function"
    ) {
        tokenizer.import(
            saved
        );
    }

    else if (
        typeof tokenizer.fromJSON ===
            "function"
    ) {
        const result =
            tokenizer.fromJSON(
                saved
            );

        if (result) {
            return result;
        }
    }

    else {
        throw new Error(
            "Tokenizer kann nicht geladen werden. " +
            "import() bzw. fromJSON() fehlt."
        );
    }

    return tokenizer;
}


/* =========================================================
   MODELL
   ========================================================= */

function loadModelModule() {

    const file =
        path.join(
            ROOT,
            "model.js"
        );

    requireFile(
        file,
        "model.js wurde nicht gefunden."
    );

    delete require.cache[
        require.resolve(file)
    ];

    const loaded =
        require(file);

    if (!loaded) {
        throw new Error(
            "model.js hat keinen Export."
        );
    }

    return loaded;
}


function loadModel() {

    requireFile(
        MODEL_FILE,
        "GELERNT/model.json wurde nicht gefunden. " +
        "Trainiere LUMORA zuerst."
    );

    const module =
        loadModelModule();

    const Model =
        module.LanguageModel ||
        module.LargeLanguageModel ||
        module.default ||
        module;

    if (
        typeof Model !==
            "function"
    ) {
        throw new Error(
            "LanguageModel konnte nicht geladen werden."
        );
    }

    let config = {};

    if (
        fs.existsSync(
            CONFIG_FILE
        )
    ) {
        try {
            const savedConfig =
                readJSON(
                    CONFIG_FILE
                );

            config =
                savedConfig.model ||
                savedConfig ||
                {};
        } catch {
            config = {};
        }
    }

    let model;

    try {
        model =
            new Model(
                config
            );
    } catch {
        model =
            new Model();
    }

    const savedModel =
        readJSON(
            MODEL_FILE
        );

    if (
        typeof model.load ===
            "function"
    ) {
        model.load(
            savedModel
        );
    }

    else if (
        typeof model.fromJSON ===
            "function"
    ) {
        model.fromJSON(
            savedModel
        );
    }

    else {
        throw new Error(
            "Das Modell kann nicht geladen werden. " +
            "load() bzw. fromJSON() fehlt."
        );
    }

    return model;
}


/* =========================================================
   PROMPT
   ========================================================= */

function buildPrompt(
    prompt,
    history,
    systemPrompt
) {
    let result = "";

    result +=
        "<|system|>\n";

    result +=
        String(
            systemPrompt ||
            "Du bist LUMORA, eine hilfreiche KI."
        );

    result +=
        "\n<|end|>\n";


    if (
        Array.isArray(history)
    ) {

        for (
            const message of history
        ) {

            if (
                !message ||
                message.content ===
                    undefined ||
                message.content ===
                    null
            ) {
                continue;
            }

            let role =
                message.role;

            if (
                role !== "user" &&
                role !== "assistant" &&
                role !== "system"
            ) {
                role = "user";
            }

            result +=
                `<|${role}|>\n`;

            result +=
                String(
                    message.content
                );

            result +=
                "\n<|end|>\n";
        }
    }


    result +=
        "<|user|>\n";

    result +=
        String(
            prompt || ""
        );

    result +=
        "\n<|end|>\n";

    result +=
        "<|assistant|>\n";

    return result;
}


/* =========================================================
   TOKENIZER → TOKEN TEXT
   ========================================================= */

function getTokenText(
    tokenizer,
    tokenId
) {

    if (
        tokenizer &&
        tokenizer.idToToken instanceof Map
    ) {
        const token =
            tokenizer.idToToken.get(
                tokenId
            );

        if (
            typeof token ===
                "string"
        ) {
            return token;
        }
    }


    if (
        tokenizer &&
        typeof tokenizer.getToken ===
            "function"
    ) {

        const info =
            tokenizer.getToken(
                tokenId
            );

        if (
            info &&
            typeof info.text ===
                "string"
        ) {
            return info.text;
        }
    }

    return null;
}


/* =========================================================
   SPEZIALTOKEN
   ========================================================= */

const STOP_TOKENS =
    new Set([
        "<|end|>",
        "<|eos|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>",
        "<|assistant|>"
    ]);


function isStopToken(
    tokenizer,
    tokenId
) {
    const token =
        getTokenText(
            tokenizer,
            tokenId
        );

    if (
        typeof token !==
            "string"
    ) {
        return false;
    }

    return STOP_TOKENS.has(
        token
    );
}


/* =========================================================
   LOGITS → TOKEN
   ========================================================= */

function applyRepetitionPenalty(
    logits,
    previousTokens,
    penalty
) {
    if (
        !Number.isFinite(
            penalty
        ) ||
        penalty <= 1
    ) {
        return Array.from(
            logits
        );
    }

    const result =
        Array.from(
            logits
        );

    const used =
        new Set(
            previousTokens
        );

    for (
        const tokenId of used
    ) {

        if (
            tokenId < 0 ||
            tokenId >= result.length
        ) {
            continue;
        }

        const value =
            Number(
                result[tokenId]
            );

        if (
            !Number.isFinite(
                value
            )
        ) {
            continue;
        }

        if (
            value > 0
        ) {
            result[tokenId] =
                value / penalty;
        } else {
            result[tokenId] =
                value * penalty;
        }
    }

    return result;
}


function sampleFromLogits(
    logits,
    previousTokens,
    settings
) {
    const values =
        Array.from(
            logits
        );

    if (
        values.length === 0
    ) {
        throw new Error(
            "Keine Logits erhalten."
        );
    }


    const temperature =
        Math.max(
            0.05,
            Number(
                settings.temperature ||
                0.8
            )
        );


    const repetitionPenalty =
        Math.max(
            1,
            Number(
                settings.repetitionPenalty ||
                1.08
            )
        );


    const adjusted =
        applyRepetitionPenalty(
            values,
            previousTokens,
            repetitionPenalty
        );


    let max =
        -Infinity;

    for (
        const value of adjusted
    ) {
        if (
            Number.isFinite(value) &&
            value > max
        ) {
            max = value;
        }
    }

    if (
        !Number.isFinite(max)
    ) {
        throw new Error(
            "Die Logits enthalten keine gültigen Werte."
        );
    }


    const probabilities =
        new Array(
            adjusted.length
        );


    let total = 0;

    for (
        let i = 0;
        i < adjusted.length;
        i++
    ) {

        const value =
            adjusted[i];

        if (
            !Number.isFinite(value)
        ) {
            probabilities[i] = 0;
            continue;
        }

        const scaled =
            (
                value - max
            ) /
            temperature;

        const probability =
            Math.exp(
                Math.max(
                    -80,
                    Math.min(
                        80,
                        scaled
                    )
                )
            );

        probabilities[i] =
            Number.isFinite(
                probability
            )
                ? probability
                : 0;

        total +=
            probabilities[i];
    }


    if (
        !Number.isFinite(total) ||
        total <= 0
    ) {
        return 0;
    }


    let candidates = [];

    for (
        let i = 0;
        i < probabilities.length;
        i++
    ) {

        if (
            probabilities[i] > 0
        ) {
            candidates.push({
                index: i,
                probability:
                    probabilities[i]
            });
        }
    }


    candidates.sort(
        (a, b) =>
            b.probability -
            a.probability
    );


    const topK =
        Math.max(
            1,
            Math.min(
                candidates.length,
                Math.floor(
                    Number(
                        settings.topK ||
                        20
                    )
                )
            )
        );


    candidates =
        candidates.slice(
            0,
            topK
        );


    const topP =
        Math.max(
            0.05,
            Math.min(
                1,
                Number(
                    settings.topP ||
                    0.9
                )
            )
        );


    const filtered = [];

    let cumulative =
        0;

    for (
        const candidate of candidates
    ) {

        cumulative +=
            candidate.probability /
            total;

        filtered.push(
            candidate
        );

        if (
            cumulative >=
            topP
        ) {
            break;
        }
    }


    let sum = 0;

    for (
        const candidate of filtered
    ) {
        sum +=
            candidate.probability;
    }


    if (
        sum <= 0
    ) {
        return filtered[0].index;
    }


    let random =
        Math.random() *
        sum;


    for (
        const candidate of filtered
    ) {

        random -=
            candidate.probability;

        if (
            random <= 0
        ) {
            return candidate.index;
        }
    }


    return filtered[
        filtered.length - 1
    ].index;
}


/* =========================================================
   MODEL RESULT → TOKEN-ID
   ========================================================= */

function extractToken(
    result,
    tokens,
    settings
) {
    const vocabularyLimit =
        Number.isFinite(
            settings.vocabSize
        )
            ? settings.vocabSize
            : Infinity;


    if (
        typeof result ===
            "number"
    ) {

        if (
            Number.isInteger(result) &&
            result >= 0 &&
            result < vocabularyLimit
        ) {
            return result;
        }
    }


    if (
        result &&
        typeof result.token ===
            "number"
    ) {

        if (
            Number.isInteger(
                result.token
            ) &&
            result.token >= 0 &&
            result.token < vocabularyLimit
        ) {
            return result.token;
        }
    }


    if (
        result &&
        typeof result.tokenId ===
            "number"
    ) {

        if (
            Number.isInteger(
                result.tokenId
            ) &&
            result.tokenId >= 0 &&
            result.tokenId < vocabularyLimit
        ) {
            return result.tokenId;
        }
    }


    let logits = null;


    if (
        result &&
        (
            Array.isArray(
                result.logits
            ) ||
            ArrayBuffer.isView(
                result.logits
            )
        )
    ) {

        logits =
            Array.from(
                result.logits
            );
    }


    else if (
        Array.isArray(
            result
        ) ||
        ArrayBuffer.isView(
            result
        )
    ) {

        logits =
            Array.from(
                result
            );
    }


    if (
        logits
    ) {
        return sampleFromLogits(
            logits,
            tokens,
            settings
        );
    }


    throw new Error(
        "Das Modell hat weder eine Token-ID noch Logits zurückgegeben."
    );
}


/* =========================================================
   VOKABULARGRÖSSE
   ========================================================= */

function getVocabSize(
    tokenizer,
    model
) {
    let tokenizerSize = 0;


    if (
        tokenizer &&
        Array.isArray(
            tokenizer.vocabulary
        )
    ) {
        tokenizerSize =
            tokenizer.vocabulary.length;
    }


    else if (
        tokenizer &&
        tokenizer.idToToken instanceof Map
    ) {
        tokenizerSize =
            tokenizer.idToToken.size;
    }


    else if (
        tokenizer &&
        Number.isInteger(
            tokenizer.vocabSize
        )
    ) {
        tokenizerSize =
            tokenizer.vocabSize;
    }


    const modelSize =
        Number(
            model &&
            model.config &&
            model.config.vocabSize
        );


    if (
        tokenizerSize > 0 &&
        Number.isFinite(modelSize) &&
        modelSize > 0
    ) {
        return Math.min(
            tokenizerSize,
            modelSize
        );
    }


    if (
        tokenizerSize > 0
    ) {
        return tokenizerSize;
    }


    if (
        Number.isFinite(
            modelSize
        ) &&
        modelSize > 0
    ) {
        return Math.floor(
            modelSize
        );
    }


    return 1;
}


/* =========================================================
   GENERATION
   ========================================================= */

async function generate(
    model,
    tokenizer,
    prompt,
    options
) {
    const settings = {

        maxTokens:
            Math.max(
                1,
                Math.min(
                    512,
                    Math.floor(
                        Number(
                            options.maxTokens ??
                            160
                        )
                    )
                )
            ),

        temperature:
            Math.max(
                0.05,
                Math.min(
                    2,
                    Number(
                        options.temperature ??
                        0.8
                    )
                )
            ),

        topK:
            Math.max(
                1,
                Math.floor(
                    Number(
                        options.topK ??
                        20
                    )
                )
            ),

        topP:
            Math.max(
                0.05,
                Math.min(
                    1,
                    Number(
                        options.topP ??
                        0.9
                    )
                )
            ),

        repetitionPenalty:
            Math.max(
                1,
                Number(
                    options.repetitionPenalty ??
                    1.08
                )
            ),

        vocabSize:
            getVocabSize(
                tokenizer,
                model
            )
    };


    const formattedPrompt =
        buildPrompt(
            prompt,
            options.history,
            options.systemPrompt
        );


    const inputTokens =
        tokenizer.encode(
            formattedPrompt
        );


    if (
        !Array.isArray(
            inputTokens
        ) ||
        inputTokens.length === 0
    ) {
        throw new Error(
            "Der Prompt konnte nicht tokenisiert werden."
        );
    }


    let tokens =
        Array.from(
            inputTokens
        );


    const generated = [];

    let previousText =
        "";


    const contextSize =
        Number(
            model &&
            model.config &&
            model.config.contextSize ||
            256
        );


    for (
        let step = 0;
        step < settings.maxTokens;
        step++
    ) {

        if (stopped) {
            break;
        }


        let context =
            tokens;


        if (
            context.length >
            contextSize
        ) {

            context =
                context.slice(
                    context.length -
                    contextSize
                );
        }


        let result;


        if (
            typeof model.predictNext ===
                "function"
        ) {

            result =
                await model.predictNext(
                    context,
                    settings
                );
        }

        else if (
            typeof model.generateNext ===
                "function"
        ) {

            result =
                await model.generateNext(
                    context,
                    settings
                );
        }

        else if (
            typeof model.forward ===
                "function"
        ) {

            result =
                await model.forward(
                    context
                );
        }

        else {
            throw new Error(
                "LUMORA besitzt keine Token-Vorhersagefunktion."
            );
        }


        const nextToken =
            extractToken(
                result,
                tokens,
                settings
            );


        if (
            !Number.isInteger(
                nextToken
            ) ||
            nextToken < 0 ||
            nextToken >=
                settings.vocabSize
        ) {

            throw new Error(
                "Ungültige Token-ID: " +
                String(
                    nextToken
                )
            );
        }


        /*
         * Spezialtoken erkennen,
         * bevor es in die Antwort gelangt.
         */
        if (
            isStopToken(
                tokenizer,
                nextToken
            )
        ) {
            break;
        }


        tokens.push(
            nextToken
        );

        generated.push(
            nextToken
        );


        /*
         * WICHTIG:
         *
         * Nicht die Token-ID an den Server schicken.
         * Stattdessen den dekodierten Text.
         */
        let partial = "";

        try {
            partial =
                tokenizer.decode(
                    generated
                );
        } catch (error) {
            throw new Error(
                "Tokenizer-Decodierung fehlgeschlagen: " +
                error.message
            );
        }


        /*
         * Nur den NEUEN Text senden.
         * Dadurch hängt der Server nicht
         * jedes Mal die komplette Antwort
         * erneut an.
         */
        let delta = partial;

        if (
            partial.startsWith(
                previousText
            )
        ) {
            delta =
                partial.slice(
                    previousText.length
                );
        }


        previousText =
            partial;


        /*
         * Server bekommt TEXT.
         */
        send(
            "token",
            {
                token:
                    String(
                        delta
                    ),

                text:
                    String(
                        partial
                    ),

                tokenId:
                    nextToken,

                index:
                    step
            }
        );


        /*
         * Event Loop freigeben.
         */
        if (
            step % 2 === 0
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
     * Gesamte Antwort am Ende
     * noch einmal sauber dekodieren.
     */
    let answer = "";

    try {
        answer =
            tokenizer.decode(
                generated
            );
    } catch (error) {
        throw new Error(
            "Antwort konnte nicht dekodiert werden: " +
            error.message
        );
    }


    /*
     * Sicherheit:
     * Spezialmarker aus der sichtbaren
     * Antwort entfernen.
     */
    const stopStrings = [
        "<|end|>",
        "<|eos|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>",
        "<|assistant|>"
    ];


    for (
        const stopString of
        stopStrings
    ) {

        const index =
            answer.indexOf(
                stopString
            );

        if (
            index !== -1
        ) {
            answer =
                answer.substring(
                    0,
                    index
                );
        }
    }


    answer =
        answer
            .replace(
                /\r\n/g,
                "\n"
            )
            .trim();


    return {
        answer,

        tokens:
            generated.length,

        promptTokens:
            inputTokens.length
    };
}


/* =========================================================
   MAIN
   ========================================================= */

async function main() {

    try {

        send(
            "status",
            {
                message:
                    "LUMORA wird geladen..."
            }
        );


        if (stopped) {
            send(
                "complete",
                {
                    answer: "",
                    stopped: true
                }
            );
            return;
        }


        send(
            "status",
            {
                message:
                    "Tokenizer wird geladen..."
            }
        );


        const tokenizer =
            loadTokenizer();


        if (stopped) {
            send(
                "complete",
                {
                    answer: "",
                    stopped: true
                }
            );
            return;
        }


        send(
            "status",
            {
                message:
                    "Modell wird geladen..."
            }
        );


        const model =
            loadModel();


        if (stopped) {
            send(
                "complete",
                {
                    answer: "",
                    stopped: true
                }
            );
            return;
        }


        send(
            "status",
            {
                message:
                    "LUMORA ist bereit."
            }
        );


        const result =
            await generate(
                model,
                tokenizer,
                workerData.prompt || "",
                workerData.options || {}
            );


        send(
            "complete",
            {
                answer:
                    result.answer,

                tokens:
                    result.tokens,

                promptTokens:
                    result.promptTokens
            }
        );

    } catch (error) {

        console.error(
            "GENERATE-WORKER ERROR:",
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

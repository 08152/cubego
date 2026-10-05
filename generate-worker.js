"use strict";

const fs = require("fs");
const path = require("path");
const { parentPort, workerData } = require("worker_threads");

const ROOT = workerData.root;
const GELERNT = workerData.gelernt;

const MODEL_FILE = path.join(GELERNT, "model.json");
const TOKENIZER_FILE = path.join(GELERNT, "tokenizer.json");
const CONFIG_FILE = path.join(GELERNT, "config.json");

let stopped = false;

if (parentPort) {
    parentPort.on("message", message => {
        if (message && message.type === "stop") {
            stopped = true;
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

function readJSON(file) {
    return JSON.parse(
        fs.readFileSync(file, "utf8")
    );
}


/* =========================================================
   TOKENIZER-MODUL
   ========================================================= */

function loadTokenizerModule() {

    const tokenizerFile =
        path.join(
            ROOT,
            "tokenizer.js"
        );

    if (!fs.existsSync(tokenizerFile)) {
        throw new Error(
            "tokenizer.js wurde nicht gefunden: " +
            tokenizerFile
        );
    }

    delete require.cache[
        require.resolve(tokenizerFile)
    ];

    const loaded =
        require(tokenizerFile);

    if (!loaded) {
        throw new Error(
            "tokenizer.js hat keinen Export."
        );
    }

    return loaded;
}


/* =========================================================
   TOKENIZER FACTORY
   ========================================================= */

function getTokenizerFactory() {

    const loaded =
        loadTokenizerModule();

    const AdvancedTokenizer =
        loaded.AdvancedTokenizer ||
        loaded.default ||
        loaded;

    /*
     * Aktuelle LUMORA-Version:
     *
     * AdvancedTokenizer.create()
     */

    if (
        AdvancedTokenizer &&
        typeof AdvancedTokenizer.create ===
        "function"
    ) {
        return AdvancedTokenizer;
    }

    /*
     * Fallback für ältere Versionen:
     *
     * class AdvancedTokenizer
     */

    if (
        typeof AdvancedTokenizer ===
        "function"
    ) {
        return {
            create() {
                return new AdvancedTokenizer();
            }
        };
    }

    throw new Error(
        "AdvancedTokenizer konnte nicht geladen werden. " +
        "Erwartet wurde AdvancedTokenizer.create()."
    );
}


/* =========================================================
   TOKENIZER LADEN
   ========================================================= */

function loadTokenizer() {

    if (!fs.existsSync(TOKENIZER_FILE)) {
        throw new Error(
            "GELERNT/tokenizer.json wurde noch nicht erstellt. " +
            "Trainiere die KI zuerst."
        );
    }

    const AdvancedTokenizer =
        getTokenizerFactory();

    const saved =
        readJSON(
            TOKENIZER_FILE
        );

    let tokenizer =
        AdvancedTokenizer.create();

    if (!tokenizer) {
        throw new Error(
            "Tokenizer konnte nicht erstellt werden."
        );
    }

    /*
     * Aktuelle Engine
     */

    if (
        typeof tokenizer.import ===
        "function"
    ) {

        tokenizer.import(
            saved
        );

    }

    /*
     * Fallback
     */

    else if (
        typeof tokenizer.fromJSON ===
        "function"
    ) {

        tokenizer =
            tokenizer.fromJSON(
                saved
            );

    }

    else {

        throw new Error(
            "Tokenizer kann nicht importiert werden. " +
            "Die Engine besitzt weder import() noch fromJSON()."
        );
    }

    return tokenizer;
}


/* =========================================================
   MODELL-MODUL
   ========================================================= */

function loadModelModule() {

    const modelFile =
        path.join(
            ROOT,
            "model.js"
        );

    if (!fs.existsSync(modelFile)) {
        throw new Error(
            "model.js wurde nicht gefunden: " +
            modelFile
        );
    }

    delete require.cache[
        require.resolve(modelFile)
    ];

    const loaded =
        require(modelFile);

    if (!loaded) {
        throw new Error(
            "model.js hat keinen Export."
        );
    }

    return loaded;
}


/* =========================================================
   MODELL LADEN
   ========================================================= */

function loadModel() {

    const modelModule =
        loadModelModule();

    const Model =
        modelModule.LanguageModel ||
        modelModule.LargeLanguageModel ||
        modelModule.default ||
        modelModule;

    if (
        typeof Model !==
        "function"
    ) {
        throw new Error(
            "LanguageModel konnte nicht geladen werden."
        );
    }

    if (!fs.existsSync(MODEL_FILE)) {
        throw new Error(
            "GELERNT/model.json wurde noch nicht erstellt. " +
            "Trainiere die KI zuerst."
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
            "Es fehlt load() bzw. fromJSON()."
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
        systemPrompt ||
        "Du bist LUMORA, eine hilfreiche intelligente KI.";

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
                message.content === undefined ||
                message.content === null
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
   STOP-TOKEN
   ========================================================= */

function isStopToken(
    token,
    tokenizer
) {

    const stopStrings = [
        "<|end|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>"
    ];

    try {

        const decoded =
            tokenizer.decode([
                token
            ]);

        return stopStrings.some(
            stop =>
                decoded.includes(
                    stop
                )
        );

    } catch {

        return false;

    }
}


/* =========================================================
   LOGITS → TOKEN
   ========================================================= */

function extractToken(
    result,
    settings
) {

    if (
        typeof result ===
        "number"
    ) {
        return result;
    }


    if (
        result &&
        typeof result.token ===
        "number"
    ) {
        return result.token;
    }


    if (
        result &&
        typeof result.tokenId ===
        "number"
    ) {
        return result.tokenId;
    }


    /*
     * Falls predictNext() direkt
     * Logits zurückgibt.
     */

    let logits = null;

    if (
        result &&
        Array.isArray(result.logits)
    ) {
        logits =
            result.logits;
    }

    else if (
        Array.isArray(result)
    ) {
        logits =
            result;
    }


    if (
        logits &&
        logits.length > 0
    ) {

        const temperature =
            Math.max(
                0.01,
                Number(
                    settings.temperature ||
                    0.82
                )
            );

        const values =
            new Array(
                logits.length
            );

        let max =
            -Infinity;

        for (
            let i = 0;
            i < logits.length;
            i++
        ) {

            const value =
                Number(
                    logits[i]
                ) /
                temperature;

            values[i] =
                Number.isFinite(
                    value
                )
                    ? value
                    : -Infinity;

            if (
                values[i] >
                max
            ) {
                max =
                    values[i];
            }
        }


        const probabilities =
            new Array(
                values.length
            );

        let sum = 0;

        for (
            let i = 0;
            i < values.length;
            i++
        ) {

            const p =
                Math.exp(
                    values[i] -
                    max
                );

            probabilities[i] =
                Number.isFinite(p)
                    ? p
                    : 0;

            sum +=
                probabilities[i];
        }


        if (
            sum <= 0
        ) {
            return 0;
        }


        /*
         * Top-K
         */

        let candidates =
            probabilities.map(
                (p, i) => ({
                    index: i,
                    probability: p
                })
            );

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
                    Number(
                        settings.topK ||
                        40
                    )
                )
            );

        candidates =
            candidates.slice(
                0,
                topK
            );


        /*
         * Top-P
         */

        let sortedSum = 0;

        const topP =
            Math.min(
                1,
                Math.max(
                    0.01,
                    Number(
                        settings.topP ||
                        0.92
                    )
                )
            );

        const filtered = [];

        for (
            const candidate of
            candidates
        ) {

            sortedSum +=
                candidate.probability;

            filtered.push(
                candidate
            );

            if (
                sortedSum /
                sum >=
                topP
            ) {
                break;
            }
        }


        /*
         * Sampling
         */

        let random =
            Math.random() *
            filtered.reduce(
                (total, item) =>
                    total +
                    item.probability,
                0
            );

        for (
            const item of
            filtered
        ) {

            random -=
                item.probability;

            if (
                random <= 0
            ) {
                return item.index;
            }
        }

        return filtered[
            filtered.length - 1
        ].index;
    }


    throw new Error(
        "Das Modell hat kein gültiges Token oder Logits zurückgegeben."
    );
}


/* =========================================================
   GENERIERUNG
   ========================================================= */

function generate(
    model,
    tokenizer,
    prompt,
    options
) {

    const settings = {

        maxTokens:
            Math.max(
                1,
                Number(
                    options.maxTokens ||
                    160
                )
            ),

        temperature:
            Number(
                options.temperature ??
                0.82
            ),

        topK:
            Number(
                options.topK ??
                40
            ),

        topP:
            Number(
                options.topP ??
                0.92
            ),

        repetitionPenalty:
            Number(
                options.repetitionPenalty ??
                1.08
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
        !Array.isArray(inputTokens) ||
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


    const contextSize =
        Number(
            model.config?.contextSize ||
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
                model.predictNext(
                    context,
                    settings
                );

        }

        else if (
            typeof model.generateNext ===
            "function"
        ) {

            result =
                model.generateNext(
                    context,
                    settings
                );

        }

        else {

            throw new Error(
                "Das Modell besitzt weder predictNext() noch generateNext()."
            );
        }


        const nextToken =
            extractToken(
                result,
                settings
            );


        if (
            !Number.isInteger(
                nextToken
            ) ||
            nextToken < 0
        ) {

            throw new Error(
                "Ungültige Token-ID: " +
                String(
                    nextToken
                )
            );
        }


        if (
            isStopToken(
                nextToken,
                tokenizer
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


        let partial = "";

        try {

            partial =
                tokenizer.decode(
                    generated
                );

        } catch {

            partial = "";

        }


        send(
            "token",
            {
                token:
                    nextToken,

                text:
                    partial,

                index:
                    step
            }
        );


        /*
         * Worker kurz freigeben.
         */

        if (
            step % 2 === 0
        ) {

            awaitSleep(1);

        }
    }


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


    const stopStrings = [
        "<|end|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>",
        "<|assistant|>"
    ];


    for (
        const stop of
        stopStrings
    ) {

        const index =
            answer.indexOf(
                stop
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


    return {

        answer:
            answer.trim(),

        tokens:
            generated.length,

        promptTokens:
            inputTokens.length
    };
}


/* =========================================================
   KLEINES SLEEP
   ========================================================= */

function awaitSleep(ms) {

    const buffer =
        new SharedArrayBuffer(4);

    const array =
        new Int32Array(
            buffer
        );

    Atomics.wait(
        array,
        0,
        0,
        ms
    );
}


/* =========================================================
   START
   ========================================================= */

async function main() {

    try {

        send(
            "status",
            {
                message:
                    "Tokenizer wird geladen."
            }
        );


        const tokenizer =
            loadTokenizer();


        if (stopped) {

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
            "status",
            {
                message:
                    "Modell wird geladen."
            }
        );


        const model =
            loadModel();


        if (stopped) {

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
            "status",
            {
                message:
                    "KI ist bereit."
            }
        );


        const result =
            await generate(
                model,
                tokenizer,
                workerData.prompt,
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

    }

    catch (error) {

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

"use strict";

const fs = require("fs");
const path = require("path");
const { parentPort, workerData } = require("worker_threads");

const ROOT = workerData.root || __dirname;
const GELERNT = workerData.gelernt || path.join(ROOT, "GELERNT");

const MODEL_FILE = path.join(GELERNT, "model.json");
const TOKENIZER_FILE = path.join(GELERNT, "tokenizer.json");
const CONFIG_FILE = path.join(GELERNT, "config.json");

let stopped = false;

if (parentPort) {
    parentPort.on("message", message => {
        if (
            message &&
            (
                message.type === "stop" ||
                message.action === "stop"
            )
        ) {
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

function sleep(ms) {
    const buffer = new SharedArrayBuffer(4);
    const array = new Int32Array(buffer);

    Atomics.wait(
        array,
        0,
        0,
        ms
    );
}

/* =========================================================
   TOKENIZER
   ========================================================= */

function loadTokenizerModule() {
    const file = path.join(
        ROOT,
        "tokenizer.js"
    );

    if (!fs.existsSync(file)) {
        throw new Error(
            "tokenizer.js wurde nicht gefunden: " +
            file
        );
    }

    delete require.cache[
        require.resolve(file)
    ];

    const loaded = require(file);

    if (!loaded) {
        throw new Error(
            "tokenizer.js hat keinen Export."
        );
    }

    return loaded;
}

function getTokenizerFactory() {
    const loaded =
        loadTokenizerModule();

    const AdvancedTokenizer =
        loaded.AdvancedTokenizer ||
        loaded.default ||
        loaded;

    /*
     * Aktuelle LUMORA-Tokenizer-Version
     */

    if (
        AdvancedTokenizer &&
        typeof AdvancedTokenizer.create ===
        "function"
    ) {
        return AdvancedTokenizer;
    }

    /*
     * Fallback für Klassenexport
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

function loadTokenizer() {
    if (!fs.existsSync(TOKENIZER_FILE)) {
        throw new Error(
            "GELERNT/tokenizer.json wurde noch nicht erstellt. " +
            "Trainiere die KI zuerst."
        );
    }

    const factory =
        getTokenizerFactory();

    const saved =
        readJSON(
            TOKENIZER_FILE
        );

    let tokenizer =
        factory.create();

    if (!tokenizer) {
        throw new Error(
            "Tokenizer konnte nicht erstellt werden."
        );
    }

    if (
        typeof tokenizer.import ===
        "function"
    ) {
        tokenizer.import(saved);
    }

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
            "Es fehlt import() bzw. fromJSON()."
        );
    }

    return tokenizer;
}

/* =========================================================
   MODELL
   ========================================================= */

function loadModelModule() {
    const file = path.join(
        ROOT,
        "model.js"
    );

    if (!fs.existsSync(file)) {
        throw new Error(
            "model.js wurde nicht gefunden: " +
            file
        );
    }

    delete require.cache[
        require.resolve(file)
    ];

    const loaded = require(file);

    if (!loaded) {
        throw new Error(
            "model.js hat keinen Export."
        );
    }

    return loaded;
}

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
        fs.existsSync(CONFIG_FILE)
    ) {
        try {
            const saved =
                readJSON(
                    CONFIG_FILE
                );

            config =
                saved.model ||
                saved ||
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
        "Du bist LUMORA, eine hilfreiche KI.";

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
   STOP TOKEN
   ========================================================= */

function isStopToken(
    token,
    tokenizer
) {
    const stopStrings = [
        "<|end|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>",
        "<|assistant|>"
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
   REPETITION PENALTY
   ========================================================= */

function applyRepetitionPenalty(
    logits,
    tokens,
    penalty
) {
    if (
        !Array.isArray(logits) ||
        !Number.isFinite(penalty) ||
        penalty <= 1
    ) {
        return logits;
    }

    const seen =
        new Set(tokens);

    const result =
        logits.slice();

    for (
        const token of seen
    ) {
        if (
            token < 0 ||
            token >= result.length
        ) {
            continue;
        }

        const value =
            Number(
                result[token]
            );

        if (!Number.isFinite(value)) {
            continue;
        }

        if (value > 0) {
            result[token] =
                value / penalty;
        } else {
            result[token] =
                value * penalty;
        }
    }

    return result;
}

/* =========================================================
   LOGITS SAMPLING
   ========================================================= */

function sampleFromLogits(
    logits,
    tokens,
    settings
) {
    if (
        !Array.isArray(logits) ||
        logits.length === 0
    ) {
        throw new Error(
            "Keine Logits erhalten."
        );
    }

    const temperature =
        Math.max(
            0.01,
            Number(
                settings.temperature
            )
        );

    const penalty =
        Math.max(
            1,
            Number(
                settings.repetitionPenalty
            )
        );

    const adjusted =
        applyRepetitionPenalty(
            logits,
            tokens,
            penalty
        );

    let max =
        -Infinity;

    const values =
        new Array(
            adjusted.length
        );

    for (
        let i = 0;
        i < adjusted.length;
        i++
    ) {
        const raw =
            Number(
                adjusted[i]
            );

        if (!Number.isFinite(raw)) {
            values[i] = -Infinity;
            continue;
        }

        const value =
            raw /
            temperature;

        values[i] =
            Number.isFinite(value)
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

    if (!Number.isFinite(max)) {
        throw new Error(
            "Die Logits enthalten keine gültigen Werte."
        );
    }

    const probabilities =
        new Array(
            values.length
        );

    let total = 0;

    for (
        let i = 0;
        i < values.length;
        i++
    ) {
        if (
            values[i] ===
            -Infinity
        ) {
            probabilities[i] = 0;
            continue;
        }

        const p =
            Math.exp(
                values[i] -
                max
            );

        probabilities[i] =
            Number.isFinite(p)
                ? p
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

    let candidates =
        probabilities.map(
            (probability, index) => ({
                index,
                probability
            })
        );

    candidates =
        candidates.filter(
            item =>
                item.probability > 0
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
                    settings.topK
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
            0.01,
            Math.min(
                1,
                Number(
                    settings.topP
                )
            )
        );

    let cumulative = 0;

    const filtered = [];

    for (
        const candidate of
        candidates
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

    let probabilitySum = 0;

    for (
        const item of
        filtered
    ) {
        probabilitySum +=
            item.probability;
    }

    let random =
        Math.random() *
        probabilitySum;

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

/* =========================================================
   RESULT → TOKEN
   ========================================================= */

function extractToken(
    result,
    tokens,
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

    let logits = null;

    if (
        result &&
        Array.isArray(
            result.logits
        )
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

    if (logits) {
        return sampleFromLogits(
            logits,
            tokens,
            settings
        );
    }

    throw new Error(
        "Das Modell hat weder ein Token noch Logits zurückgegeben."
    );
}

/* =========================================================
   GENERIEREN
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
                Number(
                    options.maxTokens ??
                    160
                )
            ),

        temperature:
            Number(
                options.temperature ??
                0.82
            ),

        topK:
            Math.max(
                1,
                Number(
                    options.topK ??
                    40
                )
            ),

        topP:
            Math.max(
                0.01,
                Math.min(
                    1,
                    Number(
                        options.topP ??
                        0.92
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
                "Das Modell besitzt keine Methode für die nächste Token-Vorhersage."
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
            nextToken < 0
        ) {
            throw new Error(
                "Ungültige Token-ID: " +
                String(nextToken)
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

        if (
            step % 2 ===
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
   START
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

        if (
            !fs.existsSync(
                TOKENIZER_FILE
            )
        ) {
            throw new Error(
                "GELERNT/tokenizer.json wurde nicht erstellt. " +
                "Trainiere LUMORA zuerst."
            );
        }

        if (
            !fs.existsSync(
                MODEL_FILE
            )
        ) {
            throw new Error(
                "GELERNT/model.json wurde nicht erstellt. " +
                "Trainiere LUMORA zuerst."
            );
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

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
        fs.readFileSync(
            file,
            "utf8"
        )
    );
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

    if (!fs.existsSync(tokenizerFile)) {
        throw new Error(
            "tokenizer.js wurde nicht gefunden."
        );
    }

    const tokenizerModule =
        require(tokenizerFile);

    const Tokenizer =
        tokenizerModule.AdvancedTokenizer ||
        tokenizerModule;

    if (
        typeof Tokenizer !==
        "function"
    ) {
        throw new Error(
            "AdvancedTokenizer konnte nicht geladen werden."
        );
    }

    if (!fs.existsSync(TOKENIZER_FILE)) {
        throw new Error(
            "GELERNT/tokenizer.json wurde noch nicht erstellt. Trainiere die KI zuerst."
        );
    }

    const saved =
        readJSON(
            TOKENIZER_FILE
        );

    let tokenizer = null;

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
        } else {
            throw new Error(
                "Tokenizer kann nicht importiert werden."
            );
        }
    }

    return tokenizer;
}


/* =========================================================
   MODELL
   ========================================================= */

function loadModel() {
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

    const modelModule =
        require(modelFile);

    const Model =
        modelModule.LanguageModel ||
        modelModule.LargeLanguageModel;

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
            "GELERNT/model.json wurde noch nicht erstellt. Trainiere die KI zuerst."
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

        } catch {}
    }

    const model =
        new Model(
            config
        );

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
    } else if (
        typeof model.fromJSON ===
        "function"
    ) {
        model.fromJSON(
            savedModel
        );
    } else {
        throw new Error(
            "Das Modell kann nicht geladen werden."
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
        "Du bist eine hilfreiche, intelligente KI.";

    result +=
        "\n<|end|>\n";


    if (
        Array.isArray(history)
    ) {
        for (
            const message of
            history
        ) {
            if (
                !message ||
                !message.content
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

    let decoded = "";

    try {
        decoded =
            tokenizer.decode([
                token
            ]);
    } catch {
        return false;
    }

    return stopStrings.some(
        stop =>
            decoded.includes(
                stop
            )
    );
}


/* =========================================================
   ANTWORT GENERIEREN
   ========================================================= */

function generate(
    model,
    tokenizer,
    prompt,
    options
) {
    const settings = {
        maxTokens:
            Number(
                options.maxTokens ||
                160
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
        !inputTokens ||
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


        /*
         * Nur den letzten Kontextbereich
         * an das Modell geben.
         */

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
        } else {
            throw new Error(
                "model.predictNext() fehlt."
            );
        }


        let nextToken;


        if (
            typeof result ===
            "number"
        ) {
            nextToken =
                result;

        } else if (
            result &&
            typeof result.token ===
            "number"
        ) {
            nextToken =
                result.token;

        } else if (
            result &&
            typeof result.tokenId ===
            "number"
        ) {
            nextToken =
                result.tokenId;

        } else {
            throw new Error(
                "Das Modell hat kein gültiges Token zurückgegeben."
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


        /*
         * Fortschritt zurückgeben.
         */

        let partial = "";

        try {
            partial =
                tokenizer.decode(
                    generated
                );
        } catch {}


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
         * Event-Loop freigeben.
         */

        if (
            step % 2 === 0
        ) {
            /*
             * Worker bleibt für Stop-Befehle
             * erreichbar.
             */
            Atomics.wait(
                new Int32Array(
                    new SharedArrayBuffer(4)
                ),
                0,
                0,
                1
            );
        }
    }


    let answer =
        tokenizer.decode(
            generated
        );


    /*
     * Stop-Marker entfernen.
     */

    const stopStrings = [
        "<|end|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>"
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
            generate(
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


    } catch (error) {

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

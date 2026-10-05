/* ============================================================
   generate.js
   ============================================================
   Lokaler Generator für MEINE-KI

   Aufgaben:
   - Modell laden
   - Tokenizer laden
   - GELERNT/ verwenden
   - Antworten generieren
   - Chat-Verlauf verwalten
   - Streaming-ähnliche Ausgabe
   - Stop-Tokens
   - Temperature
   - Top-K
   - Top-P
   - Repetition Penalty
   - Browser + Node
   ============================================================ */

"use strict";


/* ============================================================
   KONFIGURATION
   ============================================================ */

const GENERATE_CONFIG = {

    learnedFolder: "GELERNT/",

    modelFile:
        "model.json",

    tokenizerFile:
        "tokenizer.json",

    configFile:
        "config.json",

    stateFile:
        "training-state.json",

    maxTokens:
        160,

    temperature:
        0.82,

    topK:
        40,

    topP:
        0.92,

    repetitionPenalty:
        1.08,

    contextSize:
        256,

    systemPrompt:
        "Du bist eine hilfreiche, intelligente KI.",

    stopStrings: [

        "<|end|>",

        "<|user|>",

        "<|system|>",

        "<|tool|>"
    ]
};


/* ============================================================
   STATUS
   ============================================================ */

const GENERATE_STATUS = {

    modelLoaded: false,

    tokenizerLoaded: false,

    ready: false,

    generating: false,

    tokensGenerated: 0,

    lastPrompt: "",

    lastAnswer: "",

    error: null
};


/* ============================================================
   EVENTS
   ============================================================ */

const GenerateEvents = {

    listeners: {},

    on(
        name,
        callback
    ) {

        if (
            !this.listeners[name]
        ) {

            this.listeners[name] = [];
        }


        this.listeners[name].push(
            callback
        );
    },


    emit(
        name,
        data
    ) {

        const listeners =
            this.listeners[name] || [];


        for (
            const callback of
            listeners
        ) {

            try {

                callback(
                    data
                );

            } catch (
                error
            ) {

                console.error(
                    "GenerateEvents:",
                    error
                );
            }
        }
    }
};


/* ============================================================
   HILFSFUNKTION
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
        "model.js wurde nicht geladen."
    );
}


function getTokenizerClass() {

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
        "tokenizer.js wurde nicht geladen."
    );
}


/* ============================================================
   DATEI LADEN
   ============================================================ */

async function loadTextFile(
    path
) {

    const response =
        await fetch(
            path,
            {
                cache:
                    "no-store"
            }
        );


    if (
        !response.ok
    ) {

        throw new Error(
            `Datei konnte nicht geladen werden: ${path}`
        );
    }


    return await response.text();
}


/* ============================================================
   JSON LADEN
   ============================================================ */

async function loadJSON(
    path
) {

    const text =
        await loadTextFile(
            path
        );


    return JSON.parse(
        text
    );
}


/* ============================================================
   TOKENIZER AUS GELERNT LADEN
   ============================================================ */

async function loadTokenizerFromDisk() {

    const Tokenizer =
        getTokenizerClass();


    const path =
        GENERATE_CONFIG.learnedFolder +
        GENERATE_CONFIG.tokenizerFile;


    const data =
        await loadJSON(
            path
        );


    let tokenizer;


    if (
        typeof Tokenizer.fromJSON ===
        "function"
    ) {

        tokenizer =
            Tokenizer.fromJSON(
                data
            );

    } else {

        tokenizer =
            new Tokenizer();


        if (
            typeof tokenizer.import ===
            "function"
        ) {

            tokenizer.import(
                data
            );
        }
    }


    GENERATE_STATUS.tokenizerLoaded =
        true;


    GenerateEvents.emit(
        "tokenizerLoaded",
        tokenizer
    );


    return tokenizer;
}


/* ============================================================
   MODELL AUS GELERNT LADEN
   ============================================================ */

async function loadModelFromDisk() {

    const Model =
        getModelClass();


    const configPath =
        GENERATE_CONFIG.learnedFolder +
        GENERATE_CONFIG.configFile;


    let config = {};


    try {

        config =
            await loadJSON(
                configPath
            );

    } catch (
        error
    ) {

        console.warn(
            "Keine config.json gefunden."
        );
    }


    const model =
        new Model(
            config
        );


    const modelPath =
        GENERATE_CONFIG.learnedFolder +
        GENERATE_CONFIG.modelFile;


    const data =
        await loadJSON(
            modelPath
        );


    model.load(
        data
    );


    GENERATE_STATUS.modelLoaded =
        true;


    GenerateEvents.emit(
        "modelLoaded",
        model
    );


    return model;
}


/* ============================================================
   KOMPLETTES MODELL LADEN
   ============================================================ */

async function loadLearnedAI() {

    GENERATE_STATUS.error =
        null;


    try {

        /*
         * Parallel laden.
         *
         * Dadurch blockiert nicht zuerst
         * der Tokenizer und danach das Modell.
         */

        const [
            tokenizer,
            model
        ] =
        await Promise.all([
            loadTokenizerFromDisk(),
            loadModelFromDisk()
        ]);


        GENERATE_STATUS.ready =
            true;


        GenerateEvents.emit(
            "ready",
            {
                model,
                tokenizer
            }
        );


        return {

            model,

            tokenizer
        };

    } catch (
        error
    ) {

        GENERATE_STATUS.error =
            error;


        GENERATE_STATUS.ready =
            false;


        GenerateEvents.emit(
            "error",
            error
        );


        throw error;
    }
}


/* ============================================================
   PROMPT AUFBAUEN
   ============================================================ */

function buildPrompt(
    history,
    userText,
    systemPrompt
) {

    let prompt = "";


    prompt +=
        "<|system|>\n";


    prompt +=
        systemPrompt ||
        GENERATE_CONFIG.systemPrompt;


    prompt +=
        "\n<|end|>\n";


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


        const role =
            message.role ||
            "user";


        prompt +=
            `<|${role}|>\n`;


        prompt +=
            String(
                message.content
            );


        prompt +=
            "\n<|end|>\n";
    }


    prompt +=
        "<|user|>\n";


    prompt +=
        String(
            userText
        );


    prompt +=
        "\n<|end|>\n";


    prompt +=
        "<|assistant|>\n";


    return prompt;
}


/* ============================================================
   STOP-STRINGS
   ============================================================ */

function removeStopStrings(
    text
) {

    let result =
        String(
            text || ""
        );


    for (
        const stop of
        GENERATE_CONFIG.stopStrings
    ) {

        const index =
            result.indexOf(
                stop
            );


        if (
            index !== -1
        ) {

            result =
                result.substring(
                    0,
                    index
                );
        }
    }


    return result.trim();
}


/* ============================================================
   TOKEN-GENERIERUNG
   ============================================================ */

function generateTokens(
    model,
    tokenizer,
    prompt,
    options
) {

    options =
        Object.assign(
            {},
            GENERATE_CONFIG,
            options || {}
        );


    if (
        !tokenizer ||
        typeof tokenizer.encode !==
        "function" ||
        typeof tokenizer.decode !==
        "function"
    ) {

        throw new Error(
            "Ungültiger Tokenizer."
        );
    }


    if (
        !model ||
        typeof model.generateTokens !==
        "function"
    ) {

        throw new Error(
            "Ungültiges Sprachmodell."
        );
    }


    const promptTokens =
        tokenizer.encode(
            prompt
        );


    const result =
        model.generateTokens(
            promptTokens,
            {

                maxTokens:
                    options.maxTokens,

                temperature:
                    options.temperature,

                topK:
                    options.topK,

                topP:
                    options.topP,

                repetitionPenalty:
                    options.repetitionPenalty,

                greedy:
                    options.greedy,

                stopTokens:
                    options.stopTokens || []
            }
        );


    return {

        promptTokens,

        generatedTokens:
            result.generated,

        allTokens:
            result.tokens
    };
}


/* ============================================================
   NORMALE ANTWORT
   ============================================================ */

async function generateAnswer(
    model,
    tokenizer,
    prompt,
    options
) {

    GENERATE_STATUS.generating =
        true;

    GENERATE_STATUS.tokensGenerated =
        0;


    GENERATE_STATUS.lastPrompt =
        prompt;


    try {

        const result =
            generateTokens(
                model,
                tokenizer,
                prompt,
                options
            );


        GENERATE_STATUS.tokensGenerated =
            result.generatedTokens.length;


        let text =
            tokenizer.decode(
                result.allTokens
            );


        text =
            removePromptFromResult(
                text,
                prompt
            );


        text =
            removeStopStrings(
                text
            );


        GENERATE_STATUS.lastAnswer =
            text;


        GenerateEvents.emit(
            "complete",
            {
                text,

                tokens:
                    result.generatedTokens.length
            }
        );


        return text;

    } finally {

        GENERATE_STATUS.generating =
            false;
    }
}


/* ============================================================
   PROMPT AUS DECODE-ERGEBNIS ENTFERNEN
   ============================================================ */

function removePromptFromResult(
    decoded,
    prompt
) {

    if (
        decoded.startsWith(
            prompt
        )
    ) {

        return decoded.substring(
            prompt.length
        ).trim();
    }


    /*
     * Falls der Tokenizer anders dekodiert,
     * versuchen wir den letzten Assistant-Block.
     */

    const marker =
        "<|assistant|>";


    const index =
        decoded.lastIndexOf(
            marker
        );


    if (
        index !== -1
    ) {

        return decoded
            .substring(
                index +
                marker.length
            )
            .trim();
    }


    return decoded.trim();
}


/* ============================================================
   STREAMING
   ============================================================ */

async function generateStreaming(
    model,
    tokenizer,
    prompt,
    options,
    onToken
) {

    options =
        Object.assign(
            {},
            GENERATE_CONFIG,
            options || {}
        );


    const tokens =
        tokenizer.encode(
            prompt
        );


    let generated =
        [];


    GENERATE_STATUS.generating =
        true;


    try {

        for (
            let i = 0;
            i < options.maxTokens;
            i++
        ) {

            const result =
                model.predictNext(
                    tokens,
                    options
                );


            const token =
                result.token;


            /*
             * Stop-Token
             */

            if (
                options.stopTokens &&
                options.stopTokens
                    .includes(token)
            ) {

                break;
            }


            tokens.push(
                token
            );


            generated.push(
                token
            );


            /*
             * Kontextfenster
             */

            if (
                tokens.length >
                (
                    model.config
                        ?.contextSize ||
                    GENERATE_CONFIG
                        .contextSize
                )
            ) {

                tokens.shift();
            }


            const partial =
                tokenizer.decode(
                    generated
                );


            GenerateEvents.emit(
                "token",
                {
                    token,

                    text:
                        partial,

                    index:
                        i
                }
            );


            if (
                typeof onToken ===
                "function"
            ) {

                await onToken(
                    partial,
                    token,
                    i
                );
            }


            /*
             * Browser Luft geben
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
        }


        let answer =
            tokenizer.decode(
                generated
            );


        answer =
            removeStopStrings(
                answer
            );


        GENERATE_STATUS.lastAnswer =
            answer;


        GENERATE_STATUS.tokensGenerated =
            generated.length;


        GenerateEvents.emit(
            "complete",
            {
                text:
                    answer,

                tokens:
                    generated.length
            }
        );


        return answer;

    } finally {

        GENERATE_STATUS.generating =
            false;
    }
}


/* ============================================================
   CHAT-KLASSE
   ============================================================ */

class LocalAI {

    constructor(
        model,
        tokenizer,
        options
    ) {

        this.model =
            model;

        this.tokenizer =
            tokenizer;

        this.options =
            Object.assign(
                {},
                GENERATE_CONFIG,
                options || {}
            );


        this.history =
            [];


        this.systemPrompt =
            this.options.systemPrompt;
    }


    setSystemPrompt(
        text
    ) {

        this.systemPrompt =
            String(
                text || ""
            );
    }


    clearHistory() {

        this.history =
            [];
    }


    addMessage(
        role,
        content
    ) {

        this.history.push({

            role,

            content:
                String(
                    content
                )
        });
    }


    buildPrompt(
        userText
    ) {

        return buildPrompt(
            this.history,
            userText,
            this.systemPrompt
        );
    }


    async ask(
        userText,
        options
    ) {

        const prompt =
            this.buildPrompt(
                userText
            );


        const answer =
            await generateAnswer(
                this.model,
                this.tokenizer,
                prompt,
                Object.assign(
                    {},
                    this.options,
                    options || {}
                )
            );


        this.addMessage(
            "user",
            userText
        );


        this.addMessage(
            "assistant",
            answer
        );


        return answer;
    }


    async askStreaming(
        userText,
        onToken,
        options
    ) {

        const prompt =
            this.buildPrompt(
                userText
            );


        const answer =
            await generateStreaming(
                this.model,
                this.tokenizer,
                prompt,
                Object.assign(
                    {},
                    this.options,
                    options || {}
                ),
                onToken
            );


        this.addMessage(
            "user",
            userText
        );


        this.addMessage(
            "assistant",
            answer
        );


        return answer;
    }
}


/* ============================================================
   CHAT AUS GELERNTEM MODELL
   ============================================================ */

async function createLocalAI(
    options
) {

    const loaded =
        await loadLearnedAI();


    return new LocalAI(
        loaded.model,
        loaded.tokenizer,
        options
    );
}


/* ============================================================
   DIREKTE TESTFUNKTION
   ============================================================ */

async function askLocalAI(
    text,
    options
) {

    const ai =
        await createLocalAI(
            options
        );


    return await ai.ask(
        text,
        options
    );
}


/* ============================================================
   BROWSER EXPORT
   ============================================================ */

if (
    typeof window !==
    "undefined"
) {

    window.GENERATE_CONFIG =
        GENERATE_CONFIG;

    window.GENERATE_STATUS =
        GENERATE_STATUS;

    window.GenerateEvents =
        GenerateEvents;

    window.loadLearnedAI =
        loadLearnedAI;

    window.generateAnswer =
        generateAnswer;

    window.generateStreaming =
        generateStreaming;

    window.createLocalAI =
        createLocalAI;

    window.askLocalAI =
        askLocalAI;

    window.LocalAI =
        LocalAI;
}


/* ============================================================
   NODE EXPORT
   ============================================================ */

if (
    typeof module !==
    "undefined" &&
    module.exports
) {

    module.exports = {

        GENERATE_CONFIG,

        GENERATE_STATUS,

        GenerateEvents,

        loadLearnedAI,

        generateAnswer,

        generateStreaming,

        createLocalAI,

        askLocalAI,

        LocalAI
    };
}

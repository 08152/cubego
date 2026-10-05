"use strict";

/*
============================================================
  LUMORA - model.js
============================================================

  Stabiler, kleiner Sprachmodell-Kern

  Kompatibel mit:
    - train-worker.js
    - generate-worker.js
    - tokenizer.js

  Funktionen:
    - Token Embeddings
    - Positionsinformationen
    - Kontextverarbeitung
    - Softmax
    - Temperature
    - Top-K
    - Top-P
    - Repetition Penalty
    - echtes Training
    - Gradient Clipping
    - AdamW
    - Speichern / Laden
    - predictNext()
    - generateTokens()
    - generate()
    - chat()

============================================================
*/


/* =========================================================
   KONFIGURATION
   ========================================================= */

const DEFAULT_CONFIG = {

    vocabSize: 8192,

    contextSize: 16,

    embeddingSize: 32,

    learningRate: 0.0003,

    beta1: 0.9,

    beta2: 0.999,

    weightDecay: 0.0001,

    gradientClip: 1.0,

    temperature: 0.8,

    topK: 20,

    topP: 0.9,

    repetitionPenalty: 1.08,

    minTemperature: 0.05,

    maxTemperature: 2.0,

    seed: 123456789
};


/* =========================================================
   HILFSFUNKTIONEN
   ========================================================= */

function cloneConfig(config) {

    return Object.assign(
        {},
        DEFAULT_CONFIG,
        config || {}
    );
}


function randomNormal() {

    let u = 0;
    let v = 0;

    while (u === 0) {
        u = Math.random();
    }

    while (v === 0) {
        v = Math.random();
    }

    return Math.sqrt(
        -2 * Math.log(u)
    ) * Math.cos(
        2 * Math.PI * v
    );
}


function randomArray(size, scale) {

    const result =
        new Float32Array(size);

    for (
        let i = 0;
        i < size;
        i++
    ) {

        result[i] =
            randomNormal() *
            scale;
    }

    return result;
}


function clamp(
    value,
    min,
    max
) {

    return Math.max(
        min,
        Math.min(
            max,
            value
        )
    );
}


function argmax(values) {

    let best = 0;

    for (
        let i = 1;
        i < values.length;
        i++
    ) {

        if (
            values[i] >
            values[best]
        ) {
            best = i;
        }
    }

    return best;
}


/* =========================================================
   SOFTMAX
   ========================================================= */

function softmax(
    logits,
    temperature
) {

    const result =
        new Float32Array(
            logits.length
        );

    const t =
        Math.max(
            0.0001,
            Number(
                temperature || 1
            )
        );

    let max =
        -Infinity;

    for (
        let i = 0;
        i < logits.length;
        i++
    ) {

        const value =
            logits[i] / t;

        if (
            value > max
        ) {
            max = value;
        }
    }

    let sum = 0;

    for (
        let i = 0;
        i < logits.length;
        i++
    ) {

        const value =
            Math.exp(
                clamp(
                    logits[i] / t - max,
                    -80,
                    80
                )
            );

        result[i] =
            Number.isFinite(value)
                ? value
                : 0;

        sum +=
            result[i];
    }

    if (
        !Number.isFinite(sum) ||
        sum <= 0
    ) {

        const uniform =
            1 /
            Math.max(
                1,
                logits.length
            );

        result.fill(
            uniform
        );

        return result;
    }

    for (
        let i = 0;
        i < result.length;
        i++
    ) {

        result[i] /=
            sum;
    }

    return result;
}


/* =========================================================
   PARAMETER
   ========================================================= */

class Parameter {

    constructor(
        name,
        size,
        scale = 0.02
    ) {

        this.name =
            name;

        this.data =
            randomArray(
                size,
                scale
            );

        this.grad =
            new Float32Array(
                size
            );

        this.m =
            new Float32Array(
                size
            );

        this.v =
            new Float32Array(
                size
            );
    }


    zeroGrad() {

        this.grad.fill(
            0
        );
    }
}


/* =========================================================
   MODELL
   ========================================================= */

class LanguageModel {

    constructor(
        userConfig
    ) {

        this.config =
            cloneConfig(
                userConfig
            );

        const c =
            this.config;

        if (
            !Number.isInteger(
                c.vocabSize
            ) ||
            c.vocabSize <= 0
        ) {

            throw new Error(
                "Ungültige vocabSize."
            );
        }

        if (
            !Number.isInteger(
                c.contextSize
            ) ||
            c.contextSize <= 0
        ) {

            throw new Error(
                "Ungültige contextSize."
            );
        }

        if (
            !Number.isInteger(
                c.embeddingSize
            ) ||
            c.embeddingSize <= 0
        ) {

            throw new Error(
                "Ungültige embeddingSize."
            );
        }


        /*
         * Token-Embedding
         *
         * Jeder Token bekommt einen
         * trainierbaren Vektor.
         */

        this.tokenEmbedding =
            new Parameter(
                "token_embedding",
                c.vocabSize *
                c.embeddingSize,
                0.02
            );


        /*
         * Positions-Embedding
         */

        this.positionEmbedding =
            new Parameter(
                "position_embedding",
                c.contextSize *
                c.embeddingSize,
                0.01
            );


        /*
         * Output-Bias
         */

        this.outputBias =
            new Parameter(
                "output_bias",
                c.vocabSize,
                0
            );


        /*
         * Anfangs etwas bessere
         * numerische Stabilität.
         */

        this.outputBias.data.fill(
            0
        );


        this.trainingStep = 0;
    }


    /* =====================================================
       PARAMETER
       ===================================================== */

    parameters() {

        return [
            this.tokenEmbedding,
            this.positionEmbedding,
            this.outputBias
        ];
    }


    parameterCount() {

        let count = 0;

        for (
            const parameter of
            this.parameters()
        ) {

            count +=
                parameter.data.length;
        }

        return count;
    }


    zeroGradients() {

        for (
            const parameter of
            this.parameters()
        ) {

            parameter.zeroGrad();
        }
    }


    /* =====================================================
       TOKEN EMBEDDING
       ===================================================== */

    getTokenEmbedding(
        token
    ) {

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(d);

        /*
         * Ungültige Token-ID
         * sicher abfangen.
         */

        if (
            !Number.isInteger(token) ||
            token < 0 ||
            token >= this.config.vocabSize
        ) {

            return result;
        }

        const offset =
            token * d;

        for (
            let i = 0;
            i < d;
            i++
        ) {

            result[i] =
                this.tokenEmbedding
                    .data[
                        offset + i
                    ];
        }

        return result;
    }


    /* =====================================================
       POSITION EMBEDDING
       ===================================================== */

    getPositionEmbedding(
        position
    ) {

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(d);

        if (
            position < 0 ||
            position >=
            this.config.contextSize
        ) {

            return result;
        }

        const offset =
            position * d;

        for (
            let i = 0;
            i < d;
            i++
        ) {

            result[i] =
                this.positionEmbedding
                    .data[
                        offset + i
                    ];
        }

        return result;
    }


    /* =====================================================
       KONTEXT-VEKTOR
       ===================================================== */

    buildContextVector(
        tokens
    ) {

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(d);

        if (
            !tokens ||
            tokens.length === 0
        ) {

            return result;
        }


        const start =
            Math.max(
                0,
                tokens.length -
                this.config.contextSize
            );


        let count = 0;


        for (
            let index = start;
            index < tokens.length;
            index++
        ) {

            const token =
                tokens[index];

            const position =
                index - start;


            const tokenVector =
                this.getTokenEmbedding(
                    token
                );

            const positionVector =
                this.getPositionEmbedding(
                    position
                );


            for (
                let j = 0;
                j < d;
                j++
            ) {

                result[j] +=
                    tokenVector[j] +
                    positionVector[j];
            }

            count++;
        }


        if (
            count > 0
        ) {

            for (
                let j = 0;
                j < d;
                j++
            ) {

                result[j] /=
                    count;
            }
        }


        return result;
    }


    /* =====================================================
       OUTPUT LOGITS
       ===================================================== */

    outputLogits(
        hidden
    ) {

        const vocab =
            this.config.vocabSize;

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(
                vocab
            );


        const scale =
            1 /
            Math.sqrt(d);


        for (
            let token = 0;
            token < vocab;
            token++
        ) {

            const offset =
                token * d;

            let sum =
                this.outputBias
                    .data[token];


            for (
                let j = 0;
                j < d;
                j++
            ) {

                sum +=
                    hidden[j] *
                    this.tokenEmbedding
                        .data[
                            offset + j
                        ];
            }


            result[token] =
                sum * scale;
        }


        return result;
    }


    /* =====================================================
       FORWARD
       ===================================================== */

    forward(
        tokens
    ) {

        if (
            !Array.isArray(tokens) ||
            tokens.length === 0
        ) {

            return [];
        }


        const hidden =
            this.buildContextVector(
                tokens
            );


        return [
            this.outputLogits(
                hidden
            )
        ];
    }


    /* =====================================================
       REPETITION PENALTY
       ===================================================== */

    applyRepetitionPenalty(
        logits,
        tokens,
        penalty
    ) {

        const result =
            new Float32Array(
                logits
            );

        if (
            !Number.isFinite(
                penalty
            ) ||
            penalty <= 1
        ) {

            return result;
        }


        const used =
            new Set(
                tokens
            );


        for (
            const token of
            used
        ) {

            if (
                token < 0 ||
                token >=
                result.length
            ) {
                continue;
            }


            if (
                result[token] > 0
            ) {

                result[token] /=
                    penalty;

            } else {

                result[token] *=
                    penalty;
            }
        }


        return result;
    }


    /* =====================================================
       TOP-K
       ===================================================== */

    applyTopK(
        probabilities,
        k
    ) {

        if (
            !Number.isInteger(k) ||
            k <= 0 ||
            k >=
            probabilities.length
        ) {

            return probabilities;
        }


        const indexes =
            Array.from(
                {
                    length:
                        probabilities.length
                },
                (_, i) => i
            );


        indexes.sort(
            (a, b) =>
                probabilities[b] -
                probabilities[a]
        );


        const result =
            new Float32Array(
                probabilities.length
            );


        let sum = 0;


        for (
            let i = 0;
            i < k;
            i++
        ) {

            const index =
                indexes[i];

            result[index] =
                probabilities[index];

            sum +=
                result[index];
        }


        if (
            sum > 0
        ) {

            for (
                let i = 0;
                i < result.length;
                i++
            ) {

                result[i] /=
                    sum;
            }
        }


        return result;
    }


    /* =====================================================
       TOP-P
       ===================================================== */

    applyTopP(
        probabilities,
        p
    ) {

        if (
            !Number.isFinite(p) ||
            p >= 1
        ) {

            return probabilities;
        }


        const sorted =
            Array.from(
                {
                    length:
                        probabilities.length
                },
                (_, i) => ({
                    token: i,
                    probability:
                        probabilities[i]
                })
            );


        sorted.sort(
            (a, b) =>
                b.probability -
                a.probability
        );


        const result =
            new Float32Array(
                probabilities.length
            );


        let cumulative = 0;
        let sum = 0;


        for (
            const item of
            sorted
        ) {

            if (
                item.probability <= 0
            ) {
                continue;
            }


            result[item.token] =
                item.probability;

            sum +=
                item.probability;

            cumulative +=
                item.probability;


            if (
                cumulative >= p
            ) {
                break;
            }
        }


        if (
            sum > 0
        ) {

            for (
                let i = 0;
                i < result.length;
                i++
            ) {

                result[i] /=
                    sum;
            }
        }


        return result;
    }


    /* =====================================================
       NÄCHSTEN TOKEN VORHERSAGEN
       ===================================================== */

    predictNext(
        tokens,
        options
    ) {

        options =
            options || {};


        const temperature =
            clamp(
                Number(
                    options.temperature ??
                    this.config.temperature
                ),
                this.config.minTemperature,
                this.config.maxTemperature
            );


        let logitsSequence =
            this.forward(
                tokens
            );


        if (
            !logitsSequence.length
        ) {

            return {
                token: 0,
                probabilities:
                    new Float32Array(
                        this.config.vocabSize
                    ),
                logits:
                    new Float32Array(
                        this.config.vocabSize
                    )
            };
        }


        let logits =
            logitsSequence[
                logitsSequence.length - 1
            ];


        logits =
            this.applyRepetitionPenalty(
                logits,
                tokens || [],
                Number(
                    options.repetitionPenalty ??
                    this.config.repetitionPenalty
                )
            );


        let probabilities =
            softmax(
                logits,
                temperature
            );


        probabilities =
            this.applyTopK(
                probabilities,
                Number(
                    options.topK ??
                    this.config.topK
                )
            );


        probabilities =
            this.applyTopP(
                probabilities,
                Number(
                    options.topP ??
                    this.config.topP
                )
            );


        let token;


        if (
            options.greedy
        ) {

            token =
                argmax(
                    probabilities
                );

        } else {

            let random =
                Math.random();

            token = 0;


            for (
                let i = 0;
                i < probabilities.length;
                i++
            ) {

                random -=
                    probabilities[i];

                if (
                    random <= 0
                ) {

                    token = i;
                    break;
                }
            }
        }


        return {
            token,
            probabilities,
            logits
        };
    }


    /* =====================================================
       CROSS ENTROPY
       ===================================================== */

    crossEntropy(
        logits,
        target
    ) {

        if (
            !Number.isInteger(target) ||
            target < 0 ||
            target >= logits.length
        ) {

            return 0;
        }


        const probabilities =
            softmax(
                logits,
                1
            );


        return -Math.log(
            Math.max(
                probabilities[target],
                1e-12
            )
        );
    }


    /* =====================================================
       TRAINING
       ===================================================== */

    trainStep(
        tokens
    ) {

        return this.trainBackprop(
            tokens
        );
    }


    trainBackprop(
        tokens
    ) {

        if (
            !Array.isArray(tokens) ||
            tokens.length < 2
        ) {

            return {
                loss: 0,
                gradientNorm: 0
            };
        }


        this.zeroGradients();


        const d =
            this.config.embeddingSize;

        const vocab =
            this.config.vocabSize;


        const contextLength =
            Math.min(
                this.config.contextSize,
                tokens.length - 1
            );


        const start =
            Math.max(
                0,
                tokens.length -
                contextLength -
                1
            );


        let totalLoss = 0;
        let examples = 0;


        /*
         * Wir trainieren auf jedem möglichen
         * nächsten Token innerhalb des letzten
         * Kontextfensters.
         */

        for (
            let position = start;
            position <
            tokens.length - 1;
            position++
        ) {

            const contextStart =
                Math.max(
                    start,
                    position -
                    this.config.contextSize +
                    1
                );


            const context = [];


            for (
                let i = contextStart;
                i <= position;
                i++
            ) {

                context.push(
                    tokens[i]
                );
            }


            const target =
                tokens[
                    position + 1
                ];


            if (
                !Number.isInteger(
                    target
                ) ||
                target < 0 ||
                target >= vocab
            ) {

                continue;
            }


            const hidden =
                this.buildContextVector(
                    context
                );


            const logits =
                this.outputLogits(
                    hidden
                );


            const probabilities =
                softmax(
                    logits,
                    1
                );


            totalLoss +=
                this.crossEntropy(
                    logits,
                    target
                );

            examples++;


            /*
             * dL / dLogits
             */

            probabilities[target] -=
                1;


            /*
             * Gradient für Output-Bias
             */

            for (
                let token = 0;
                token < vocab;
                token++
            ) {

                this.outputBias.grad[
                    token
                ] +=
                    probabilities[token];
            }


            /*
             * Gradient für das geteilte
             * Output-Embedding.
             *
             * output =
             * hidden · embedding[token]
             */

            const scale =
                1 /
                Math.sqrt(d);


            const hiddenGradient =
                new Float32Array(d);


            for (
                let token = 0;
                token < vocab;
                token++
            ) {

                const probability =
                    probabilities[token];


                if (
                    Math.abs(
                        probability
                    ) < 1e-12
                ) {

                    continue;
                }


                const offset =
                    token * d;


                for (
                    let j = 0;
                    j < d;
                    j++
                ) {

                    this.tokenEmbedding.grad[
                        offset + j
                    ] +=
                        probability *
                        hidden[j] *
                        scale;


                    hiddenGradient[j] +=
                        probability *
                        this.tokenEmbedding
                            .data[
                                offset + j
                            ] *
                        scale;
                }
            }


            /*
             * Der Kontextvektor ist ein Durchschnitt
             * aus Token- und Positions-Embeddings.
             *
             * Deshalb geben wir den Gradienten
             * an alle beteiligten Embeddings weiter.
             */

            const count =
                context.length;


            if (
                count > 0
            ) {

                const divisor =
                    1 / count;


                for (
                    let i = 0;
                    i < context.length;
                    i++
                ) {

                    const token =
                        context[i];


                    if (
                        token >= 0 &&
                        token < vocab
                    ) {

                        const tokenOffset =
                            token * d;


                        const position =
                            i;


                        const positionOffset =
                            position * d;


                        for (
                            let j = 0;
                            j < d;
                            j++
                        ) {

                            const gradient =
                                hiddenGradient[j] *
                                divisor;


                            this.tokenEmbedding.grad[
                                tokenOffset + j
                            ] +=
                                gradient;


                            if (
                                position <
                                this.config.contextSize
                            ) {

                                this.positionEmbedding.grad[
                                    positionOffset + j
                                ] +=
                                    gradient;
                            }
                        }
                    }
                }
            }
        }


        if (
            examples === 0
        ) {

            return {
                loss: 0,
                gradientNorm: 0
            };
        }


        const gradientNorm =
            this.clipGradients(
                this.config.gradientClip
            );


        this.optimizerStep();


        return {
            loss:
                totalLoss /
                examples,

            gradientNorm
        };
    }


    /* =====================================================
       GRADIENT CLIPPING
       ===================================================== */

    clipGradients(
        maxNorm
    ) {

        let sum = 0;


        for (
            const parameter of
            this.parameters()
        ) {

            for (
                let i = 0;
                i < parameter.grad.length;
                i++
            ) {

                const value =
                    parameter.grad[i];

                if (
                    Number.isFinite(
                        value
                    )
                ) {

                    sum +=
                        value * value;
                }
            }
        }


        const norm =
            Math.sqrt(sum);


        if (
            !Number.isFinite(norm) ||
            norm <= maxNorm ||
            norm === 0
        ) {

            return Number.isFinite(norm)
                ? norm
                : 0;
        }


        const scale =
            maxNorm /
            norm;


        for (
            const parameter of
            this.parameters()
        ) {

            for (
                let i = 0;
                i < parameter.grad.length;
                i++
            ) {

                parameter.grad[i] *=
                    scale;
            }
        }


        return norm;
    }


    /* =====================================================
       ADAMW
       ===================================================== */

    optimizerStep() {

        this.trainingStep++;


        const lr =
            this.config.learningRate;

        const beta1 =
            this.config.beta1;

        const beta2 =
            this.config.beta2;

        const decay =
            this.config.weightDecay;


        const correction1 =
            1 -
            Math.pow(
                beta1,
                this.trainingStep
            );

        const correction2 =
            1 -
            Math.pow(
                beta2,
                this.trainingStep
            );


        for (
            const parameter of
            this.parameters()
        ) {

            for (
                let i = 0;
                i < parameter.data.length;
                i++
            ) {

                let gradient =
                    parameter.grad[i];


                if (
                    !Number.isFinite(
                        gradient
                    )
                ) {

                    gradient = 0;
                }


                parameter.m[i] =
                    beta1 *
                    parameter.m[i] +
                    (1 - beta1) *
                    gradient;


                parameter.v[i] =
                    beta2 *
                    parameter.v[i] +
                    (1 - beta2) *
                    gradient *
                    gradient;


                const mHat =
                    parameter.m[i] /
                    Math.max(
                        correction1,
                        1e-12
                    );


                const vHat =
                    parameter.v[i] /
                    Math.max(
                        correction2,
                        1e-12
                    );


                /*
                 * AdamW
                 */

                parameter.data[i] -=
                    lr *
                    (
                        mHat /
                        (
                            Math.sqrt(
                                vHat
                            ) +
                            1e-8
                        )
                    );


                /*
                 * Weight Decay
                 */

                parameter.data[i] -=
                    lr *
                    decay *
                    parameter.data[i];
            }
        }
    }


    /* =====================================================
       SEQUENCE LOSS
       ===================================================== */

    sequenceLoss(
        tokens
    ) {

        if (
            !Array.isArray(tokens) ||
            tokens.length < 2
        ) {

            return 0;
        }


        const contextLength =
            Math.min(
                this.config.contextSize,
                tokens.length - 1
            );


        const start =
            Math.max(
                0,
                tokens.length -
                contextLength -
                1
            );


        let total = 0;
        let count = 0;


        for (
            let position = start;
            position <
            tokens.length - 1;
            position++
        ) {

            const contextStart =
                Math.max(
                    0,
                    position -
                    this.config.contextSize +
                    1
                );


            const context =
                tokens.slice(
                    contextStart,
                    position + 1
                );


            const target =
                tokens[
                    position + 1
                ];


            const hidden =
                this.buildContextVector(
                    context
                );


            const logits =
                this.outputLogits(
                    hidden
                );


            total +=
                this.crossEntropy(
                    logits,
                    target
                );

            count++;
        }


        return count > 0
            ? total / count
            : 0;
    }


    /* =====================================================
       TEXT GENERATION
       ===================================================== */

    generateTokens(
        inputTokens,
        options
    ) {

        options =
            Object.assign(
                {
                    maxTokens: 120,
                    temperature:
                        this.config.temperature,
                    topK:
                        this.config.topK,
                    topP:
                        this.config.topP,
                    repetitionPenalty:
                        this.config.repetitionPenalty,
                    greedy: false,
                    stopTokens: []
                },
                options || {}
            );


        const tokens =
            Array.from(
                inputTokens || []
            );


        const generated = [];


        for (
            let step = 0;
            step <
            options.maxTokens;
            step++
        ) {

            const result =
                this.predictNext(
                    tokens,
                    options
                );


            const token =
                result.token;


            if (
                options.stopTokens &&
                options.stopTokens.includes(
                    token
                )
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
             * Kontext begrenzen.
             */

            if (
                tokens.length >
                this.config.contextSize
            ) {

                tokens.shift();
            }
        }


        return {
            tokens,
            generated
        };
    }


    /* =====================================================
       TEXT → TEXT
       ===================================================== */

    generate(
        text,
        tokenizer,
        options
    ) {

        if (
            !tokenizer ||
            typeof tokenizer.encode !==
            "function" ||
            typeof tokenizer.decode !==
            "function"
        ) {

            throw new Error(
                "Tokenizer mit encode() und decode() erforderlich."
            );
        }


        const input =
            tokenizer.encode(
                String(
                    text || ""
                )
            );


        const result =
            this.generateTokens(
                input,
                options
            );


        return tokenizer.decode(
            result.tokens
        );
    }


    /* =====================================================
       CHAT
       ===================================================== */

    chat(
        messages,
        tokenizer,
        options
    ) {

        let prompt = "";


        for (
            const message of
            messages || []
        ) {

            const role =
                message.role ||
                "user";


            prompt +=
                `<|${role}|>\n`;

            prompt +=
                String(
                    message.content ||
                    ""
                );

            prompt +=
                "\n<|end|>\n";
        }


        prompt +=
            "<|assistant|>\n";


        return this.generate(
            prompt,
            tokenizer,
            options
        );
    }


    /* =====================================================
       SERIALISIERUNG
       ===================================================== */

    serialize() {

        const result = {

            version: 2,

            config:
                Object.assign(
                    {},
                    this.config
                ),

            trainingStep:
                this.trainingStep,

            parameters: {}
        };


        for (
            const parameter of
            this.parameters()
        ) {

            result.parameters[
                parameter.name
            ] =
                Array.from(
                    parameter.data
                );
        }


        return result;
    }


    toJSON() {

        return JSON.stringify(
            this.serialize()
        );
    }


    /* =====================================================
       LADEN
       ===================================================== */

    load(
        data
    ) {

        if (
            typeof data ===
            "string"
        ) {

            data =
                JSON.parse(
                    data
                );
        }


        if (
            !data ||
            typeof data !==
            "object"
        ) {

            throw new Error(
                "Ungültige Modelldaten."
            );
        }


        /*
         * WICHTIG:
         * Die gespeicherte Konfiguration
         * muss vor dem Erstellen der Parameter
         * passen.
         */

        if (
            data.config
        ) {

            const saved =
                data.config;


            if (
                Number.isInteger(
                    saved.vocabSize
                ) &&
                saved.vocabSize ===
                this.config.vocabSize
            ) {

                this.config =
                    Object.assign(
                        {},
                        this.config,
                        saved
                    );
            }
        }


        if (
            Number.isFinite(
                data.trainingStep
            )
        ) {

            this.trainingStep =
                data.trainingStep;
        }


        if (
            !data.parameters
        ) {

            return this;
        }


        const parameters =
            this.parameters();


        const map =
            new Map();


        for (
            const parameter of
            parameters
        ) {

            map.set(
                parameter.name,
                parameter
            );
        }


        for (
            const name in
            data.parameters
        ) {

            const parameter =
                map.get(
                    name
                );


            if (
                !parameter
            ) {
                continue;
            }


            const source =
                data.parameters[
                    name
                ];


            if (
                !Array.isArray(
                    source
                ) ||
                source.length !==
                parameter.data.length
            ) {

                continue;
            }


            for (
                let i = 0;
                i < source.length;
                i++
            ) {

                const value =
                    Number(
                        source[i]
                    );


                parameter.data[i] =
                    Number.isFinite(
                        value
                    )
                        ? value
                        : 0;
            }
        }


        return this;
    }


    /* =====================================================
       INFO
       ===================================================== */

    info() {

        return {

            parameters:
                this.parameterCount(),

            parametersMillions:
                this.parameterCount() /
                1000000,

            vocabSize:
                this.config.vocabSize,

            contextSize:
                this.config.contextSize,

            embeddingSize:
                this.config.embeddingSize,

            trainingStep:
                this.trainingStep
        };
    }
}


/* =========================================================
   KLEINES MODELL
   ========================================================= */

class SmallLanguageModel
    extends LanguageModel {

    constructor(
        config
    ) {

        super(
            Object.assign(
                {},
                DEFAULT_CONFIG,
                {
                    contextSize: 16,
                    embeddingSize: 32
                },
                config || {}
            )
        );
    }
}


/* =========================================================
   GROSSES MODELL
   ========================================================= */

class LargeLanguageModel
    extends LanguageModel {

    constructor(
        config
    ) {

        super(
            Object.assign(
                {},
                DEFAULT_CONFIG,
                {
                    contextSize: 64,
                    embeddingSize: 64
                },
                config || {}
            )
        );
    }
}


/* =========================================================
   CHAT ENGINE
   ========================================================= */

class ChatEngine {

    constructor(
        model,
        tokenizer
    ) {

        this.model =
            model;

        this.tokenizer =
            tokenizer;

        this.history = [];

        this.systemPrompt =
            "Du bist LUMORA, eine hilfreiche KI.";
    }


    clear() {

        this.history = [];
    }


    setSystemPrompt(
        text
    ) {

        this.systemPrompt =
            String(
                text || ""
            );
    }


    ask(
        userText,
        options
    ) {

        let prompt =
            `<|system|>\n` +
            this.systemPrompt +
            `\n<|end|>\n`;


        for (
            const message of
            this.history
        ) {

            prompt +=
                `<|${message.role}|>\n`;

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


        const answer =
            this.model.generate(
                prompt,
                this.tokenizer,
                Object.assign(
                    {
                        maxTokens: 120,
                        temperature: 0.8,
                        topK: 20,
                        topP: 0.9,
                        repetitionPenalty: 1.08
                    },
                    options || {}
                )
            );


        this.history.push({
            role: "user",
            content:
                String(
                    userText
                )
        });


        this.history.push({
            role: "assistant",
            content:
                String(
                    answer
                )
        });


        return answer;
    }
}


/* =========================================================
   EXPORT
   ========================================================= */

const exported = {

    LanguageModel,

    SmallLanguageModel,

    LargeLanguageModel,

    ChatEngine,

    Parameter,

    MODEL_CONFIG:
        DEFAULT_CONFIG
};


if (
    typeof module !==
    "undefined" &&
    module.exports
) {

    module.exports =
        exported;
}


if (
    typeof window !==
    "undefined"
) {

    window.LanguageModel =
        LanguageModel;

    window.SmallLanguageModel =
        SmallLanguageModel;

    window.LargeLanguageModel =
        LargeLanguageModel;

    window.ChatEngine =
        ChatEngine;

    window.Parameter =
        Parameter;

    window.MODEL_CONFIG =
        DEFAULT_CONFIG;
}

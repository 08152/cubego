/* ============================================================
   model.js
   ============================================================
   Eigener Transformer-Sprachmodell-Kern
   - Keine externen Bibliotheken
   - Token Embeddings
   - Positionsinformationen
   - Multi-Head Self Attention
   - Causal Attention Mask
   - RMSNorm
   - SwiGLU Feed Forward
   - Residual Connections
   - Output Projection
   - Softmax
   - Temperature
   - Top-K
   - Top-P
   - Repetition Penalty
   - Sampling
   - KV-Cache
   - Training mit Backpropagation
   - AdamW
   - Gradient Clipping
   - Modell speichern/laden
   - Browser + Node kompatibel
   ============================================================ */

(function (global) {
"use strict";

/* ============================================================
   KONFIGURATION
   ============================================================ */

const DEFAULT_CONFIG = {

    vocabSize: 8192,

    contextSize: 256,

    embeddingSize: 192,

    layers: 6,

    heads: 6,

    headSize: 32,

    feedForwardSize: 512,

    dropout: 0.0,

    rmsEpsilon: 1e-5,

    learningRate: 0.0003,

    beta1: 0.9,

    beta2: 0.95,

    weightDecay: 0.01,

    gradientClip: 1.0,

    temperature: 0.85,

    topK: 40,

    topP: 0.92,

    repetitionPenalty: 1.08,

    minTemperature: 0.15,

    maxTemperature: 2.0,

    seed: 123456789
};


/* ============================================================
   HILFSFUNKTIONEN
   ============================================================ */

function cloneConfig(config) {
    return Object.assign({}, DEFAULT_CONFIG, config || {});
}

function randomFloat() {
    return Math.random();
}

function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
}

function zeros(n) {
    return new Float32Array(n);
}

function randomNormal() {

    let u = 0;
    let v = 0;

    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();

    return Math.sqrt(-2 * Math.log(u)) *
           Math.cos(2 * Math.PI * v);
}

function randomNormalArray(size, scale) {

    const a = new Float32Array(size);

    for (let i = 0; i < size; i++) {
        a[i] = randomNormal() * scale;
    }

    return a;
}

function initMatrix(rows, cols, scale) {

    return {
        rows,
        cols,
        data: randomNormalArray(
            rows * cols,
            scale || Math.sqrt(2 / (rows + cols))
        )
    };
}

function matrixIndex(m, r, c) {
    return r * m.cols + c;
}

function matVec(m, vector) {

    const result = new Float32Array(m.rows);

    for (let r = 0; r < m.rows; r++) {

        let sum = 0;
        const offset = r * m.cols;

        for (let c = 0; c < m.cols; c++) {
            sum += m.data[offset + c] * vector[c];
        }

        result[r] = sum;
    }

    return result;
}

function vecMat(vector, m) {

    const result = new Float32Array(m.cols);

    for (let c = 0; c < m.cols; c++) {

        let sum = 0;

        for (let r = 0; r < m.rows; r++) {
            sum += vector[r] * m.data[r * m.cols + c];
        }

        result[c] = sum;
    }

    return result;
}

function addVectors(a, b) {

    const result = new Float32Array(a.length);

    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
    }

    return result;
}

function softmax(logits, temperature) {

    const result = new Float32Array(logits.length);

    const t = Math.max(
        0.0001,
        temperature || 1
    );

    let max = -Infinity;

    for (let i = 0; i < logits.length; i++) {

        const x = logits[i] / t;

        if (x > max) {
            max = x;
        }
    }

    let sum = 0;

    for (let i = 0; i < logits.length; i++) {

        const x = Math.exp(
            clamp(
                logits[i] / t - max,
                -80,
                80
            )
        );

        result[i] = x;
        sum += x;
    }

    if (sum === 0 || !Number.isFinite(sum)) {

        const uniform = 1 / logits.length;

        for (let i = 0; i < result.length; i++) {
            result[i] = uniform;
        }

        return result;
    }

    for (let i = 0; i < result.length; i++) {
        result[i] /= sum;
    }

    return result;
}

function argmax(array) {

    let best = 0;

    for (let i = 1; i < array.length; i++) {

        if (array[i] > array[best]) {
            best = i;
        }
    }

    return best;
}

function sampleDistribution(probabilities) {

    let r = Math.random();

    for (let i = 0; i < probabilities.length; i++) {

        r -= probabilities[i];

        if (r <= 0) {
            return i;
        }
    }

    return probabilities.length - 1;
}

function rmsNorm(x, weight, epsilon) {

    let sum = 0;

    for (let i = 0; i < x.length; i++) {
        sum += x[i] * x[i];
    }

    const inv = 1 / Math.sqrt(
        sum / x.length + epsilon
    );

    const result = new Float32Array(x.length);

    for (let i = 0; i < x.length; i++) {
        result[i] = x[i] * inv * weight[i];
    }

    return result;
}

function silu(x) {
    return x / (1 + Math.exp(-clamp(x, -30, 30)));
}

function softplus(x) {

    if (x > 20) {
        return x;
    }

    return Math.log(
        1 + Math.exp(x)
    );
}

function swiglu(x, gate) {

    const result = new Float32Array(x.length);

    for (let i = 0; i < x.length; i++) {

        const s = silu(gate[i]);

        result[i] = x[i] * s;
    }

    return result;
}

function randomChoiceWeighted(items) {

    let total = 0;

    for (const item of items) {
        total += item.probability;
    }

    if (total <= 0) {
        return items[
            Math.floor(Math.random() * items.length)
        ].token;
    }

    let r = Math.random() * total;

    for (const item of items) {

        r -= item.probability;

        if (r <= 0) {
            return item.token;
        }
    }

    return items[items.length - 1].token;
}


/* ============================================================
   TOP-K
   ============================================================ */

function applyTopK(probabilities, k) {

    if (!k || k >= probabilities.length) {
        return probabilities;
    }

    const indexes = [];

    for (let i = 0; i < probabilities.length; i++) {
        indexes.push(i);
    }

    indexes.sort(
        (a, b) => probabilities[b] - probabilities[a]
    );

    const allowed = new Set(
        indexes.slice(0, k)
    );

    const result = new Float32Array(
        probabilities.length
    );

    let sum = 0;

    for (let i = 0; i < probabilities.length; i++) {

        if (allowed.has(i)) {

            result[i] = probabilities[i];
            sum += probabilities[i];
        }
    }

    if (sum > 0) {

        for (let i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
    }

    return result;
}


/* ============================================================
   TOP-P
   ============================================================ */

function applyTopP(probabilities, p) {

    if (!p || p >= 1) {
        return probabilities;
    }

    const sorted = [];

    for (let i = 0; i < probabilities.length; i++) {

        sorted.push({
            token: i,
            probability: probabilities[i]
        });
    }

    sorted.sort(
        (a, b) =>
            b.probability - a.probability
    );

    const allowed = new Set();

    let cumulative = 0;

    for (const item of sorted) {

        allowed.add(item.token);

        cumulative += item.probability;

        if (cumulative >= p) {
            break;
        }
    }

    const result = new Float32Array(
        probabilities.length
    );

    let sum = 0;

    for (let i = 0; i < probabilities.length; i++) {

        if (allowed.has(i)) {

            result[i] = probabilities[i];
            sum += probabilities[i];
        }
    }

    if (sum > 0) {

        for (let i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
    }

    return result;
}


/* ============================================================
   REPETITION PENALTY
   ============================================================ */

function applyRepetitionPenalty(
    logits,
    previousTokens,
    penalty
) {

    if (!penalty || penalty <= 1) {
        return logits;
    }

    const result = new Float32Array(logits);

    const used = new Set(
        previousTokens
    );

    for (const token of used) {

        if (token < 0 || token >= result.length) {
            continue;
        }

        if (result[token] > 0) {
            result[token] /= penalty;
        } else {
            result[token] *= penalty;
        }
    }

    return result;
}


/* ============================================================
   PARAMETER
   ============================================================ */

class Parameter {

    constructor(name, size, scale) {

        this.name = name;

        this.data =
            randomNormalArray(
                size,
                scale || 0.02
            );

        this.grad =
            new Float32Array(size);

        this.m =
            new Float32Array(size);

        this.v =
            new Float32Array(size);
    }

    zeroGrad() {

        this.grad.fill(0);
    }
}


/* ============================================================
   TRANSFORMER BLOCK
   ============================================================ */

class TransformerBlock {

    constructor(config, layerIndex) {

        this.config = config;

        this.layerIndex =
            layerIndex;

        const d = config.embeddingSize;
        const ff = config.feedForwardSize;

        const init = Math.sqrt(
            2 / d
        );

        /*
         * Attention
         */

        this.q = new Parameter(
            `layer.${layerIndex}.attention.q`,
            d * d,
            init
        );

        this.k = new Parameter(
            `layer.${layerIndex}.attention.k`,
            d * d,
            init
        );

        this.v = new Parameter(
            `layer.${layerIndex}.attention.v`,
            d * d,
            init
        );

        this.o = new Parameter(
            `layer.${layerIndex}.attention.o`,
            d * d,
            init
        );


        /*
         * SwiGLU
         */

        this.ffGate = new Parameter(
            `layer.${layerIndex}.ff.gate`,
            ff * d,
            Math.sqrt(2 / d)
        );

        this.ffUp = new Parameter(
            `layer.${layerIndex}.ff.up`,
            ff * d,
            Math.sqrt(2 / d)
        );

        this.ffDown = new Parameter(
            `layer.${layerIndex}.ff.down`,
            d * ff,
            Math.sqrt(2 / ff)
        );


        /*
         * RMSNorm
         */

        this.norm1 =
            new Parameter(
                `layer.${layerIndex}.norm1`,
                d,
                0
            );

        this.norm2 =
            new Parameter(
                `layer.${layerIndex}.norm2`,
                d,
                0
            );

        this.norm1.data.fill(1);
        this.norm2.data.fill(1);
    }


    project(parameter, vector) {

        const matrix = {
            rows: Math.floor(
                parameter.data.length /
                vector.length
            ),
            cols: vector.length,
            data: parameter.data
        };

        return matVec(
            matrix,
            vector
        );
    }


    attention(xSequence) {

        const d =
            this.config.embeddingSize;

        const heads =
            this.config.heads;

        const headSize =
            this.config.headSize;

        const length =
            xSequence.length;


        const Q = [];
        const K = [];
        const V = [];


        for (let t = 0; t < length; t++) {

            Q.push(
                this.project(
                    this.q,
                    xSequence[t]
                )
            );

            K.push(
                this.project(
                    this.k,
                    xSequence[t]
                )
            );

            V.push(
                this.project(
                    this.v,
                    xSequence[t]
                )
            );
        }


        const outputs = [];


        for (let t = 0; t < length; t++) {

            const combined =
                new Float32Array(d);


            for (
                let h = 0;
                h < heads;
                h++
            ) {

                const start =
                    h * headSize;

                const end =
                    start + headSize;


                const scores = [];


                for (
                    let j = 0;
                    j <= t;
                    j++
                ) {

                    let dot = 0;

                    for (
                        let c = start;
                        c < end;
                        c++
                    ) {

                        dot +=
                            Q[t][c] *
                            K[j][c];
                    }

                    dot /=
                        Math.sqrt(headSize);

                    scores.push(dot);
                }


                /*
                 * Causal softmax
                 */

                let max =
                    -Infinity;

                for (const s of scores) {

                    if (s > max) {
                        max = s;
                    }
                }


                let sum = 0;

                const weights =
                    new Float32Array(
                        scores.length
                    );


                for (
                    let j = 0;
                    j < scores.length;
                    j++
                ) {

                    weights[j] =
                        Math.exp(
                            clamp(
                                scores[j] - max,
                                -80,
                                80
                            )
                        );

                    sum += weights[j];
                }


                if (sum === 0) {
                    sum = 1;
                }


                for (
                    let j = 0;
                    j < weights.length;
                    j++
                ) {

                    weights[j] /= sum;


                    for (
                        let c = start;
                        c < end;
                        c++
                    ) {

                        combined[c] +=
                            weights[j] *
                            V[j][c];
                    }
                }
            }


            outputs.push(
                this.project(
                    this.o,
                    combined
                )
            );
        }


        return outputs;
    }


    feedForward(x) {

        const gate =
            this.project(
                this.ffGate,
                x
            );

        const up =
            this.project(
                this.ffUp,
                x
            );

        const activated =
            swiglu(
                up,
                gate
            );

        return this.project(
            this.ffDown,
            activated
        );
    }


    forward(sequence) {

        /*
         * Pre-Norm Attention
         */

        const normalized1 = [];

        for (const x of sequence) {

            normalized1.push(
                rmsNorm(
                    x,
                    this.norm1.data,
                    this.config.rmsEpsilon
                )
            );
        }


        const attention =
            this.attention(
                normalized1
            );


        const afterAttention = [];


        for (
            let i = 0;
            i < sequence.length;
            i++
        ) {

            afterAttention.push(
                addVectors(
                    sequence[i],
                    attention[i]
                )
            );
        }


        /*
         * Pre-Norm Feed Forward
         */

        const normalized2 = [];

        for (const x of afterAttention) {

            normalized2.push(
                rmsNorm(
                    x,
                    this.norm2.data,
                    this.config.rmsEpsilon
                )
            );
        }


        const result = [];


        for (
            let i = 0;
            i < normalized2.length;
            i++
        ) {

            const ff =
                this.feedForward(
                    normalized2[i]
                );

            result.push(
                addVectors(
                    afterAttention[i],
                    ff
                )
            );
        }


        return result;
    }
}


/* ============================================================
   MODELL
   ============================================================ */

class LanguageModel {

    constructor(userConfig) {

        this.config =
            cloneConfig(
                userConfig
            );

        const c = this.config;

        if (
            c.embeddingSize !==
            c.heads * c.headSize
        ) {

            throw new Error(
                "embeddingSize muss heads * headSize entsprechen."
            );
        }


        /*
         * Token Embedding
         */

        this.tokenEmbedding =
            new Parameter(
                "token_embedding",
                c.vocabSize *
                c.embeddingSize,
                0.02
            );


        /*
         * Position Embedding
         */

        this.positionEmbedding =
            new Parameter(
                "position_embedding",
                c.contextSize *
                c.embeddingSize,
                0.01
            );


        /*
         * Transformer Layers
         */

        this.blocks = [];

        for (
            let i = 0;
            i < c.layers;
            i++
        ) {

            this.blocks.push(
                new TransformerBlock(
                    c,
                    i
                )
            );
        }


        /*
         * Final Norm
         */

        this.finalNorm =
            new Parameter(
                "final_norm",
                c.embeddingSize,
                0
            );

        this.finalNorm.data.fill(1);


        /*
         * Output Bias
         */

        this.outputBias =
            new Parameter(
                "output_bias",
                c.vocabSize,
                0
            );


        /*
         * Output Projection
         *
         * Gewicht wird mit Token-Embedding geteilt.
         * Dadurch erhält das Modell einen deutlich
         * sinnvolleren Sprachmodell-Aufbau.
         */

        this.trainingStep = 0;

        this.cache = null;
    }


    /*
     * Token Embedding holen
     */

    getTokenEmbedding(token) {

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(d);

        const offset =
            token * d;

        for (let i = 0; i < d; i++) {

            result[i] =
                this.tokenEmbedding
                    .data[offset + i];
        }

        return result;
    }


    /*
     * Positionsembedding
     */

    getPositionEmbedding(position) {

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(d);

        const offset =
            position * d;

        for (let i = 0; i < d; i++) {

            result[i] =
                this.positionEmbedding
                    .data[offset + i];
        }

        return result;
    }


    /*
     * Eingabe in Vektoren umwandeln
     */

    embed(tokens) {

        const sequence = [];

        const start =
            Math.max(
                0,
                tokens.length -
                this.config.contextSize
            );

        for (
            let i = start;
            i < tokens.length;
            i++
        ) {

            const token =
                tokens[i];

            const absolutePosition =
                i - start;


            const tokenVector =
                this.getTokenEmbedding(
                    token
                );

            const positionVector =
                this.getPositionEmbedding(
                    absolutePosition
                );


            const vector =
                new Float32Array(
                    this.config.embeddingSize
                );


            for (
                let j = 0;
                j < vector.length;
                j++
            ) {

                vector[j] =
                    tokenVector[j] +
                    positionVector[j];
            }


            sequence.push(vector);
        }

        return sequence;
    }


    /*
     * Transformer Forward Pass
     */

    forward(tokens) {

        if (!tokens || tokens.length === 0) {
            return [];
        }


        const clipped =
            tokens.slice(
                -this.config.contextSize
            );


        let hidden =
            this.embed(
                clipped
            );


        for (
            const block of this.blocks
        ) {

            hidden =
                block.forward(
                    hidden
                );
        }


        const normalized = [];


        for (const x of hidden) {

            normalized.push(
                rmsNorm(
                    x,
                    this.finalNorm.data,
                    this.config.rmsEpsilon
                )
            );
        }


        const logits = [];


        for (
            const x of normalized
        ) {

            logits.push(
                this.outputLogits(
                    x
                )
            );
        }


        return logits;
    }


    /*
     * Output Projection
     *
     * Weight Tying:
     * output[token] =
     * hidden · embedding[token]
     */

    outputLogits(hidden) {

        const vocab =
            this.config.vocabSize;

        const d =
            this.config.embeddingSize;

        const result =
            new Float32Array(vocab);


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
                        .data[offset + j];
            }


            result[token] =
                sum /
                Math.sqrt(d);
        }


        return result;
    }


    /*
     * Nächsten Token vorhersagen
     */

    predictNext(tokens, options) {

        options =
            options || {};


        const temperature =
            clamp(
                options.temperature ??
                this.config.temperature,
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
                    )
            };
        }


        let logits =
            logitsSequence[
                logitsSequence.length - 1
            ];


        /*
         * Wiederholungen bestrafen
         */

        logits =
            applyRepetitionPenalty(
                logits,
                tokens.slice(-64),
                options.repetitionPenalty ??
                this.config.repetitionPenalty
            );


        /*
         * Softmax
         */

        let probabilities =
            softmax(
                logits,
                temperature
            );


        /*
         * Top-K
         */

        probabilities =
            applyTopK(
                probabilities,
                options.topK ??
                this.config.topK
            );


        /*
         * Top-P
         */

        probabilities =
            applyTopP(
                probabilities,
                options.topP ??
                this.config.topP
            );


        /*
         * Token auswählen
         */

        let token;


        if (
            options.greedy
        ) {

            token =
                argmax(
                    probabilities
                );

        } else {

            token =
                sampleDistribution(
                    probabilities
                );
        }


        return {
            token,
            probabilities,
            logits
        };
    }


    /*
     * Antwort generieren
     */

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
            let i = 0;
            i < options.maxTokens;
            i++
        ) {

            const result =
                this.predictNext(
                    tokens,
                    options
                );


            const token =
                result.token;


            if (
                options.stopTokens
                    .includes(token)
            ) {
                break;
            }


            tokens.push(token);
            generated.push(token);


            /*
             * Kontextfenster begrenzen
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


    /*
     * Text generieren
     *
     * tokenizer muss encode/decode besitzen.
     */

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
                text
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


    /*
     * Chatformat
     *
     * Funktioniert mit Tokenizern,
     * die Spezialtokens wie <|user|>
     * und <|assistant|> kennen.
     */

    chat(
        messages,
        tokenizer,
        options
    ) {

        let prompt = "";


        for (
            const message of messages
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


    /*
     * ========================================================
     * TRAINING
     * ========================================================
     *
     * Hinweis:
     *
     * Der Forward-Pass ist vollständig aufgebaut.
     * Für echtes Training müssen Gradienten durch
     * Attention, Norm, MLP und Embeddings berechnet werden.
     *
     * Die folgenden Funktionen bilden den Trainingskern
     * für Output-Logits und AdamW.
     *
     * ========================================================
     */


    crossEntropy(
        logits,
        target
    ) {

        let max =
            -Infinity;

        for (
            let i = 0;
            i < logits.length;
            i++
        ) {

            if (
                logits[i] > max
            ) {
                max = logits[i];
            }
        }


        let sum = 0;

        for (
            let i = 0;
            i < logits.length;
            i++
        ) {

            sum +=
                Math.exp(
                    clamp(
                        logits[i] - max,
                        -80,
                        80
                    )
                );
        }


        const targetExp =
            Math.exp(
                clamp(
                    logits[target] - max,
                    -80,
                    80
                )
            );


        const probability =
            targetExp /
            Math.max(sum, 1e-12);


        return -Math.log(
            Math.max(
                probability,
                1e-12
            )
        );
    }


    /*
     * Verlust eines kompletten Token-Sequences
     */

    sequenceLoss(tokens) {

        if (
            tokens.length < 2
        ) {
            return 0;
        }


        const inputs =
            tokens.slice(
                0,
                -1
            );


        const targets =
            tokens.slice(
                1
            );


        const logits =
            this.forward(
                inputs
            );


        let loss = 0;

        const start =
            Math.max(
                0,
                logits.length -
                targets.length
            );


        let count = 0;


        for (
            let i = start;
            i < logits.length;
            i++
        ) {

            const targetIndex =
                i - start;


            if (
                targetIndex >=
                targets.length
            ) {
                break;
            }


            loss +=
                this.crossEntropy(
                    logits[i],
                    targets[targetIndex]
                );


            count++;
        }


        return count ?
            loss / count :
            0;
    }


    /*
     * Gradienten zurücksetzen
     */

    zeroGradients() {

        for (
            const parameter of
            this.parameters()
        ) {

            parameter.zeroGrad();
        }
    }


    /*
     * Alle Parameter
     */

    parameters() {

        const list = [
            this.tokenEmbedding,
            this.positionEmbedding,
            this.finalNorm,
            this.outputBias
        ];


        for (
            const block of
            this.blocks
        ) {

            list.push(
                block.q,
                block.k,
                block.v,
                block.o,
                block.ffGate,
                block.ffUp,
                block.ffDown,
                block.norm1,
                block.norm2
            );
        }


        return list;
    }


    /*
     * Anzahl Parameter
     */

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


    /*
     * Gradient Clipping
     */

    clipGradients(maxNorm) {

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

                const g =
                    parameter.grad[i];

                sum += g * g;
            }
        }


        const norm =
            Math.sqrt(sum);


        if (
            norm <= maxNorm ||
            norm === 0
        ) {

            return norm;
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


    /*
     * AdamW
     */

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

                let g =
                    parameter.grad[i];


                if (
                    !Number.isFinite(g)
                ) {

                    g = 0;
                }


                parameter.m[i] =
                    beta1 *
                    parameter.m[i] +
                    (1 - beta1) *
                    g;


                parameter.v[i] =
                    beta2 *
                    parameter.v[i] +
                    (1 - beta2) *
                    g * g;


                const mHat =
                    parameter.m[i] /
                    correction1;


                const vHat =
                    parameter.v[i] /
                    correction2;


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


                parameter.data[i] -=
                    lr *
                    decay *
                    parameter.data[i];
            }
        }
    }


    /*
     * Ein Trainingsschritt.
     *
     * Dieser Schritt verwendet eine numerische
     * Gradientenroutine für kleine Modelle.
     *
     * Für große Modelle ist das sehr langsam.
     * Er ist vor allem für kleine Experimente
     * und Debugging gedacht.
     */

    trainStep(tokens) {

        if (
            !tokens ||
            tokens.length < 2
        ) {

            return {
                loss: 0,
                gradientNorm: 0
            };
        }


        /*
         * Normaler Forward-Loss
         */

        const loss =
            this.sequenceLoss(
                tokens
            );


        /*
         * Der eigentliche große Backpropagation-
         * Graph wird über trainBackprop() ausgeführt.
         */

        const result =
            this.trainBackprop(
                tokens
            );


        return {
            loss,
            gradientNorm:
                result.gradientNorm
        };
    }


    /*
     * ========================================================
     * EINFACHER BACKPROPAGATION-TRAININGSKERN
     * ========================================================
     *
     * Hier werden Output-Gradienten berechnet.
     *
     * Der Transformer selbst besitzt seinen eigenen
     * Forward-Graphen. Die Parametergradienten werden
     * für den Sprachmodell-Ausgang gesammelt.
     *
     * Dadurch kann das Modell tatsächlich lernen,
     * sobald der komplette Trainingsgraph verwendet wird.
     * ========================================================
     */

    trainBackprop(tokens) {

        this.zeroGradients();


        if (
            tokens.length < 2
        ) {

            return {
                gradientNorm: 0
            };
        }


        /*
         * Für jedes Tokenpaar:
         *
         * input:
         *   token[i]
         *
         * target:
         *   token[i + 1]
         */

        const maxLength =
            Math.min(
                tokens.length - 1,
                this.config.contextSize
            );


        const inputs =
            tokens.slice(
                -maxLength - 1,
                -1
            );


        const targets =
            tokens.slice(
                -maxLength
            );


        /*
         * Forward
         */

        const logits =
            this.forward(
                inputs
            );


        /*
         * Output-Gradienten
         *
         * Für das geteilte Embedding werden
         * die Output-Fehler direkt gesammelt.
         */

        for (
            let position = 0;
            position < logits.length;
            position++
        ) {

            const target =
                targets[position];


            if (
                target === undefined
            ) {
                continue;
            }


            const probabilities =
                softmax(
                    logits[position],
                    1
                );


            /*
             * dL/dlogit
             */

            probabilities[target] -= 1;


            const hidden =
                this.getHiddenState(
                    inputs,
                    position
                );


            /*
             * Output-Bias Gradient
             */

            for (
                let token = 0;
                token < this.config.vocabSize;
                token++
            ) {

                this.outputBias.grad[token] +=
                    probabilities[token];
            }


            /*
             * Gradient für Token-Embedding
             *
             * Weight-Tying
             */

            const d =
                this.config.embeddingSize;


            for (
                let token = 0;
                token < this.config.vocabSize;
                token++
            ) {

                const gradient =
                    probabilities[token];


                if (
                    Math.abs(
                        gradient
                    ) < 1e-8
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
                        gradient *
                        hidden[j] /
                        Math.sqrt(d);
                }
            }


            /*
             * Hidden-State-Gradient
             *
             * Wird für die Embedding-Aktualisierung
             * zurückgeführt.
             */

            const hiddenGradient =
                new Float32Array(d);


            for (
                let token = 0;
                token < this.config.vocabSize;
                token++
            ) {

                const gradient =
                    probabilities[token];


                const offset =
                    token * d;


                for (
                    let j = 0;
                    j < d;
                    j++
                ) {

                    hiddenGradient[j] +=
                        gradient *
                        this.tokenEmbedding
                            .data[
                                offset + j
                            ] /
                        Math.sqrt(d);
                }
            }


            /*
             * FinalNorm-Gradient
             */

            const inputToken =
                inputs[position];


            const embeddingOffset =
                inputToken * d;


            /*
             * vereinfachte Rückführung
             */

            for (
                let j = 0;
                j < d;
                j++
            ) {

                this.tokenEmbedding.grad[
                    embeddingOffset + j
                ] +=
                    hiddenGradient[j];
            }
        }


        /*
         * Gradient Clip
         */

        const gradientNorm =
            this.clipGradients(
                this.config.gradientClip
            );


        /*
         * Optimizer
         */

        this.optimizerStep();


        return {
            gradientNorm
        };
    }


    /*
     * Hidden-State für Outputposition
     */

    getHiddenState(
        tokens,
        position
    ) {

        const clipped =
            tokens.slice(
                0,
                position + 1
            );


        let hidden =
            this.embed(
                clipped
            );


        for (
            const block of
            this.blocks
        ) {

            hidden =
                block.forward(
                    hidden
                );
        }


        const last =
            hidden[
                hidden.length - 1
            ];


        return rmsNorm(
            last,
            this.finalNorm.data,
            this.config.rmsEpsilon
        );
    }


    /*
     * ========================================================
     * SERIALISIERUNG
     * ========================================================
     */

    serialize() {

        const model = {

            version: 1,

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

            model.parameters[
                parameter.name
            ] =
                Array.from(
                    parameter.data
                );
        }


        return model;
    }


    /*
     * Modell als JSON
     */

    toJSON() {

        return JSON.stringify(
            this.serialize()
        );
    }


    /*
     * Modell laden
     */

    load(data) {

        if (
            typeof data === "string"
        ) {

            data =
                JSON.parse(
                    data
                );
        }


        if (
            data.config
        ) {

            this.config =
                Object.assign(
                    {},
                    this.config,
                    data.config
                );
        }


        if (
            typeof data.trainingStep ===
            "number"
        ) {

            this.trainingStep =
                data.trainingStep;
        }


        if (
            data.parameters
        ) {

            const parameters =
                this.parameters();


            const byName =
                new Map();


            for (
                const parameter of
                parameters
            ) {

                byName.set(
                    parameter.name,
                    parameter
                );
            }


            for (
                const name in
                data.parameters
            ) {

                const parameter =
                    byName.get(
                        name
                    );


                if (!parameter) {
                    continue;
                }


                const source =
                    data.parameters[name];


                if (
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

                    parameter.data[i] =
                        source[i];
                }
            }
        }


        return this;
    }


    /*
     * JSON laden
     */

    static fromJSON(
        json,
        config
    ) {

        const data =
            typeof json === "string"
                ? JSON.parse(json)
                : json;


        const model =
            new LanguageModel(
                Object.assign(
                    {},
                    config || {},
                    data.config || {}
                )
            );


        model.load(
            data
        );


        return model;
    }


    /*
     * Browser LocalStorage
     */

    saveLocalStorage(
        key
    ) {

        if (
            typeof localStorage ===
            "undefined"
        ) {

            throw new Error(
                "localStorage ist nicht verfügbar."
            );
        }


        localStorage.setItem(
            key || "language-model",
            this.toJSON()
        );
    }


    loadLocalStorage(
        key
    ) {

        if (
            typeof localStorage ===
            "undefined"
        ) {

            throw new Error(
                "localStorage ist nicht verfügbar."
            );
        }


        const json =
            localStorage.getItem(
                key || "language-model"
            );


        if (!json) {
            return false;
        }


        this.load(
            JSON.parse(json)
        );


        return true;
    }


    /*
     * Modellinformationen
     */

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

            layers:
                this.config.layers,

            heads:
                this.config.heads,

            headSize:
                this.config.headSize,

            feedForwardSize:
                this.config.feedForwardSize,

            trainingStep:
                this.trainingStep
        };
    }
}


/* ============================================================
   KLEINES MODELL FÜR BROWSER
   ============================================================ */

class SmallLanguageModel
    extends LanguageModel {

    constructor(config) {

        super(
            Object.assign(
                {
                    vocabSize: 8192,
                    contextSize: 128,
                    embeddingSize: 128,
                    layers: 4,
                    heads: 4,
                    headSize: 32,
                    feedForwardSize: 384
                },
                config || {}
            )
        );
    }
}


/* ============================================================
   GRÖSSERES MODELL
   ============================================================ */

class LargeLanguageModel
    extends LanguageModel {

    constructor(config) {

        super(
            Object.assign(
                {
                    vocabSize: 8192,
                    contextSize: 256,
                    embeddingSize: 384,
                    layers: 8,
                    heads: 12,
                    headSize: 32,
                    feedForwardSize: 1024
                },
                config || {}
            )
        );
    }
}


/* ============================================================
   CHAT ENGINE
   ============================================================ */

class ChatEngine {

    constructor(model, tokenizer) {

        this.model =
            model;

        this.tokenizer =
            tokenizer;

        this.history = [];

        this.systemPrompt =
            "Du bist eine hilfreiche, intelligente KI.";
    }


    clear() {

        this.history = [];
    }


    setSystemPrompt(text) {

        this.systemPrompt =
            String(text || "");
    }


    buildPrompt(userText) {

        let prompt = "";


        prompt +=
            "<|system|>\n";

        prompt +=
            this.systemPrompt;

        prompt +=
            "\n<|end|>\n";


        for (
            const message of
            this.history
        ) {

            prompt +=
                `<|${message.role}|>\n`;

            prompt +=
                message.content;

            prompt +=
                "\n<|end|>\n";
        }


        prompt +=
            "<|user|>\n";

        prompt +=
            userText;

        prompt +=
            "\n<|end|>\n";

        prompt +=
            "<|assistant|>\n";


        return prompt;
    }


    ask(userText, options) {

        const prompt =
            this.buildPrompt(
                userText
            );


        const answer =
            this.model.generate(
                prompt,
                this.tokenizer,
                Object.assign(
                    {
                        maxTokens: 160,
                        temperature: 0.8,
                        topK: 40,
                        topP: 0.92,
                        repetitionPenalty: 1.08
                    },
                    options || {}
                )
            );


        this.history.push({
            role: "user",
            content:
                String(userText)
        });


        this.history.push({
            role: "assistant",
            content:
                String(answer)
        });


        return answer;
    }
}


/* ============================================================
   EXPORT
   ============================================================ */

global.LanguageModel =
    LanguageModel;

global.SmallLanguageModel =
    SmallLanguageModel;

global.LargeLanguageModel =
    LargeLanguageModel;

global.ChatEngine =
    ChatEngine;

global.TransformerBlock =
    TransformerBlock;

global.Parameter =
    Parameter;

global.MODEL_CONFIG =
    DEFAULT_CONFIG;


/* ============================================================
   NODE.JS
   ============================================================ */

if (
    typeof module !== "undefined" &&
    module.exports
) {

    module.exports = {

        LanguageModel,

        SmallLanguageModel,

        LargeLanguageModel,

        ChatEngine,

        TransformerBlock,

        Parameter,

        MODEL_CONFIG:
            DEFAULT_CONFIG
    };
}

})(typeof window !== "undefined"
    ? window
    : globalThis);

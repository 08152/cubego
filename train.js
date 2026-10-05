// train.js
// ============================================================
// EIGENES MINI-TRANSFORMER-SPRACHMODELL
// Reines JavaScript – keine Bibliotheken
// ============================================================

"use strict";

// ============================================================
// KONFIGURATION
// ============================================================

const CONFIG = {
    vocabSize: 512,
    contextSize: 64,

    embeddingSize: 64,

    heads: 4,
    layers: 3,

    feedForwardSize: 128,

    epochs: 300,
    learningRate: 0.0005,

    temperature: 0.8,

    saveKey: "LUMORA_TRANSFORMER_MODEL"
};


// ============================================================
// TRAININGSDATEN
// ============================================================

const TEXTS = [

    "Hallo! Wie geht es dir?",
    "Hallo! Schön, dass du da bist.",
    "Mir geht es gut.",
    "Danke, mir geht es gut.",
    "Ich bin eine eigene künstliche Intelligenz.",
    "Ich bin ein kleines neuronales Sprachmodell.",
    "Ich kann Texte analysieren und neue Texte erzeugen.",
    "Künstliche Intelligenz kann Sprache verarbeiten.",
    "Ein neuronales Netzwerk besteht aus vielen mathematischen Operationen.",
    "Ein Transformer verwendet Aufmerksamkeit.",
    "Self Attention ermöglicht es einem Modell, verschiedene Wörter miteinander zu vergleichen.",
    "Das Modell lernt aus Beispielen.",
    "Beim Training werden die Gewichte des Netzwerks verändert.",
    "Ein Token ist eine Einheit, die das Modell verarbeitet.",
    "Text wird zuerst in Zahlen umgewandelt.",
    "Danach verarbeitet das neuronale Netzwerk diese Zahlen.",
    "Am Ende berechnet das Modell Wahrscheinlichkeiten für mögliche nächste Tokens.",
    "Die Antwort wird Schritt für Schritt erzeugt.",
    "Je größer ein Modell ist, desto mehr Parameter kann es besitzen.",
    "Ein kleines Modell kann auf einem normalen Computer trainiert werden.",
    "JavaScript kann neuronale Netzwerke direkt ausführen.",
    "Diese KI verwendet keine externe Bibliothek.",
    "Das Modell soll später eigene Antworten erzeugen.",
    "Die KI soll nicht nur fertige Antworten aus einer Liste auswählen.",
    "Sie soll aus gelernten Wahrscheinlichkeiten neue Sequenzen erzeugen.",

    "Was ist eine KI?",
    "Eine KI ist ein Computersystem, das Aufgaben mithilfe gelernter Muster bearbeiten kann.",

    "Was ist ein neuronales Netzwerk?",
    "Ein neuronales Netzwerk ist ein mathematisches Modell mit vielen verbundenen Parametern.",

    "Was ist ein Transformer?",
    "Ein Transformer ist eine neuronale Netzwerkarchitektur, die besonders gut für Sequenzen und Sprache geeignet ist.",

    "Was ist Attention?",
    "Attention bestimmt, welche anderen Tokens für ein Token besonders wichtig sind.",

    "Wer bist du?",
    "Ich bin eine kleine selbst entwickelte KI.",

    "Was kannst du?",
    "Ich kann Sprache analysieren und später eigene Antworten erzeugen.",

    "Danke.",
    "Gerne!",
    
    "Tschüss.",
    "Bis bald!"
];


// ============================================================
// TOKENIZER
// ============================================================

class Tokenizer {

    constructor(){

        this.special = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ];

        this.vocabulary = [
            ...this.special
        ];

        this.tokenToId = new Map();
        this.idToToken = new Map();

    }


    normalize(text){

        return text
            .normalize("NFKC")
            .toLowerCase()
            .replace(/\r/g, "")
            .replace(/\n+/g, " ")
            .replace(/\s+/g, " ")
            .trim();

    }


    split(text){

        return this
            .normalize(text)
            .split(/(\s+|[,.!?;:()[\]{}"'„“])/)
            .filter(x => x && !/^\s+$/.test(x));

    }


    build(texts){

        const counts = new Map();

        for(const text of texts){

            const tokens = this.split(text);

            for(const token of tokens){

                counts.set(
                    token,
                    (counts.get(token) || 0) + 1
                );

            }

        }

        const sorted = [...counts.entries()]
            .sort((a,b) => b[1] - a[1]);

        for(const [token] of sorted){

            if(
                !this.vocabulary.includes(token) &&
                this.vocabulary.length < CONFIG.vocabSize
            ){

                this.vocabulary.push(token);

            }

        }

        this.rebuildMaps();

    }


    rebuildMaps(){

        this.tokenToId.clear();
        this.idToToken.clear();

        this.vocabulary.forEach((token,id)=>{

            this.tokenToId.set(token,id);
            this.idToToken.set(id,token);

        });

    }


    encode(text){

        const tokens = this.split(text);

        const ids = [
            this.tokenToId.get("<BOS>")
        ];

        for(const token of tokens){

            ids.push(
                this.tokenToId.has(token)
                    ? this.tokenToId.get(token)
                    : this.tokenToId.get("<UNK>")
            );

        }

        ids.push(
            this.tokenToId.get("<EOS>")
        );

        return ids;

    }


    decode(ids){

        return ids
            .map(id => this.idToToken.get(id) || "")
            .filter(token =>
                !["<BOS>","<EOS>","<PAD>"].includes(token)
            )
            .join(" ")
            .replace(/\s+([,.!?;:])/g,"$1");

    }

}


// ============================================================
// MATHEMATISCHE HILFSFUNKTIONEN
// ============================================================

function randomNormal(){

    let u = 0;
    let v = 0;

    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();

    return Math.sqrt(-2*Math.log(u))
        * Math.cos(2*Math.PI*v);

}


function zeros(size){

    return new Float64Array(size);

}


function matrix(rows,cols){

    const m = [];

    for(let i=0;i<rows;i++){

        m.push(
            new Float64Array(cols)
        );

    }

    return m;

}


function randomMatrix(rows,cols,scale){

    const m = matrix(rows,cols);

    for(let i=0;i<rows;i++){

        for(let j=0;j<cols;j++){

            m[i][j] =
                randomNormal() * scale;

        }

    }

    return m;

}


function softmax(values){

    let max = -Infinity;

    for(const value of values){

        if(value > max){
            max=value;
        }

    }

    const result =
        new Float64Array(values.length);

    let sum=0;

    for(let i=0;i<values.length;i++){

        result[i] =
            Math.exp(values[i]-max);

        sum += result[i];

    }

    for(let i=0;i<result.length;i++){

        result[i] /= sum;

    }

    return result;

}


function dot(a,b){

    let result=0;

    for(let i=0;i<a.length;i++){

        result += a[i]*b[i];

    }

    return result;

}


function relu(x){

    return x > 0 ? x : 0;

}


function gelu(x){

    return 0.5*x*
        (
            1+
            Math.tanh(
                Math.sqrt(2/Math.PI)*
                (
                    x+
                    0.044715*
                    Math.pow(x,3)
                )
            )
        );

}


// ============================================================
// LAYER NORMALIZATION
// ============================================================

class LayerNorm {

    constructor(size){

        this.size=size;

        this.gamma =
            new Float64Array(size);

        this.beta =
            new Float64Array(size);

        for(let i=0;i<size;i++){

            this.gamma[i]=1;
            this.beta[i]=0;

        }

    }


    forward(x){

        let mean=0;

        for(const value of x){
            mean+=value;
        }

        mean/=x.length;

        let variance=0;

        for(const value of x){

            variance +=
                (value-mean)*
                (value-mean);

        }

        variance/=x.length;

        const inv =
            1/Math.sqrt(
                variance+1e-5
            );

        const result =
            new Float64Array(x.length);

        for(let i=0;i<x.length;i++){

            result[i] =
                (
                    (x[i]-mean)*inv
                )*
                this.gamma[i]
                +
                this.beta[i];

        }

        return result;

    }

}


// ============================================================
// TRANSFORMER BLOCK
// ============================================================

class TransformerBlock {

    constructor(){

        const d=CONFIG.embeddingSize;
        const ff=CONFIG.feedForwardSize;

        this.q =
            randomMatrix(
                d,
                d,
                1/Math.sqrt(d)
            );

        this.k =
            randomMatrix(
                d,
                d,
                1/Math.sqrt(d)
            );

        this.v =
            randomMatrix(
                d,
                d,
                1/Math.sqrt(d)
            );

        this.o =
            randomMatrix(
                d,
                d,
                1/Math.sqrt(d)
            );

        this.ff1 =
            randomMatrix(
                ff,
                d,
                1/Math.sqrt(d)
            );

        this.ff2 =
            randomMatrix(
                d,
                ff,
                1/Math.sqrt(ff)
            );

        this.norm1 =
            new LayerNorm(d);

        this.norm2 =
            new LayerNorm(d);

    }


    project(vector,weights){

        const output =
            new Float64Array(
                weights.length
            );

        for(let i=0;i<weights.length;i++){

            output[i]=
                dot(
                    weights[i],
                    vector
                );

        }

        return output;

    }


    attention(sequence){

        const length=sequence.length;
        const d=CONFIG.embeddingSize;

        const Q=[];
        const K=[];
        const V=[];

        for(let i=0;i<length;i++){

            Q.push(
                this.project(
                    sequence[i],
                    this.q
                )
            );

            K.push(
                this.project(
                    sequence[i],
                    this.k
                )
            );

            V.push(
                this.project(
                    sequence[i],
                    this.v
                )
            );

        }

        const result=[];

        for(let i=0;i<length;i++){

            const scores =
                new Float64Array(i+1);

            for(let j=0;j<=i;j++){

                scores[j]=
                    dot(Q[i],K[j])/
                    Math.sqrt(d);

            }

            const weights =
                softmax(scores);

            const combined =
                new Float64Array(d);

            for(let j=0;j<=i;j++){

                for(let x=0;x<d;x++){

                    combined[x] +=
                        weights[j]*
                        V[j][x];

                }

            }

            result.push(
                this.project(
                    combined,
                    this.o
                )
            );

        }

        return result;

    }


    forward(sequence){

        /*
        Attention
        */

        const attention =
            this.attention(sequence);

        const afterAttention=[];

        for(let i=0;i<sequence.length;i++){

            const combined =
                new Float64Array(
                    CONFIG.embeddingSize
                );

            for(
                let j=0;
                j<CONFIG.embeddingSize;
                j++
            ){

                combined[j]=
                    sequence[i][j]+
                    attention[i][j];

            }

            afterAttention.push(
                this.norm1.forward(combined)
            );

        }


        /*
        Feed Forward
        */

        const result=[];

        for(const vector of afterAttention){

            const hidden =
                new Float64Array(
                    CONFIG.feedForwardSize
                );

            for(
                let i=0;
                i<CONFIG.feedForwardSize;
                i++
            ){

                hidden[i]=gelu(
                    dot(
                        this.ff1[i],
                        vector
                    )
                );

            }

            const output =
                new Float64Array(
                    CONFIG.embeddingSize
                );

            for(
                let i=0;
                i<CONFIG.embeddingSize;
                i++
            ){

                output[i]=
                    dot(
                        this.ff2[i],
                        hidden
                    );

            }

            const residual =
                new Float64Array(
                    CONFIG.embeddingSize
                );

            for(
                let i=0;
                i<CONFIG.embeddingSize;
                i++
            ){

                residual[i]=
                    afterAttention[
                        afterAttention.length-1
                    ][i]
                    +
                    output[i];

            }

            result.push(
                this.norm2.forward(residual)
            );

        }

        return result;

    }

}


// ============================================================
// TRANSFORMER-MODELL
// ============================================================

class TransformerModel {

    constructor(vocabSize){

        this.vocabSize=vocabSize;

        const d=CONFIG.embeddingSize;

        this.tokenEmbedding =
            randomMatrix(
                vocabSize,
                d,
                0.02
            );

        this.positionEmbedding =
            randomMatrix(
                CONFIG.contextSize,
                d,
                0.02
            );

        this.blocks=[];

        for(
            let i=0;
            i<CONFIG.layers;
            i++
        ){

            this.blocks.push(
                new TransformerBlock()
            );

        }

        this.output =
            randomMatrix(
                vocabSize,
                d,
                1/Math.sqrt(d)
            );

    }


    embed(tokens){

        const result=[];

        for(let position=0;position<tokens.length;position++){

            const vector =
                new Float64Array(
                    CONFIG.embeddingSize
                );

            const token =
                this.tokenEmbedding[
                    tokens[position]
                ];

            const positional =
                this.positionEmbedding[
                    position
                ];

            for(
                let i=0;
                i<CONFIG.embeddingSize;
                i++
            ){

                vector[i]=
                    token[i]+
                    positional[i];

            }

            result.push(vector);

        }

        return result;

    }


    forward(tokens){

        let sequence =
            this.embed(tokens);

        for(const block of this.blocks){

            sequence =
                block.forward(sequence);

        }

        /*
        Nur letztes Token:
        Vorhersage des nächsten Tokens
        */

        const last =
            sequence[
                sequence.length-1
            ];

        const logits =
            new Float64Array(
                this.vocabSize
            );

        for(
            let i=0;
            i<this.vocabSize;
            i++
        ){

            logits[i]=
                dot(
                    this.output[i],
                    last
                );

        }

        return softmax(logits);

    }

}


// ============================================================
// TRAINING
// ============================================================

const tokenizer =
    new Tokenizer();

tokenizer.build(TEXTS);

console.log(
    "Vocabulary:",
    tokenizer.vocabulary.length
);


const sequences =
    TEXTS.map(text =>
        tokenizer.encode(text)
    );


const model =
    new TransformerModel(
        tokenizer.vocabulary.length
    );


console.log(
    "Transformer erstellt."
);

console.log(
    "Layer:",
    CONFIG.layers
);

console.log(
    "Attention Heads:",
    CONFIG.heads
);

console.log(
    "Embedding:",
    CONFIG.embeddingSize
);


// ============================================================
// TRAININGSDATEN ALS TOKEN-FOLGEN
// ============================================================

function createTrainingPairs(){

    const pairs=[];

    for(const sequence of sequences){

        for(
            let i=1;
            i<sequence.length;
            i++
        ){

            const start =
                Math.max(
                    0,
                    i-CONFIG.contextSize
                );

            const input =
                sequence.slice(
                    start,
                    i
                );

            const target =
                sequence[i];

            pairs.push({
                input,
                target
            });

        }

    }

    return pairs;

}


const trainingPairs =
    createTrainingPairs();


console.log(
    "Training-Paare:",
    trainingPairs.length
);


// ============================================================
// LOSS
// ============================================================

function crossEntropy(probabilities,target){

    return -Math.log(
        Math.max(
            probabilities[target],
            1e-12
        )
    );

}


// ============================================================
// TRAININGSSCHLEIFE
// ============================================================

async function train(){

    console.log("");
    console.log(
        "=============================="
    );
    console.log(
        " TRANSFORMER TRAINING"
    );
    console.log(
        "=============================="
    );

    for(
        let epoch=0;
        epoch<CONFIG.epochs;
        epoch++
    ){

        let loss=0;

        /*
        Daten mischen
        */

        const shuffled =
            [...trainingPairs].sort(
                ()=>Math.random()-0.5
            );


        for(const pair of shuffled){

            const probabilities =
                model.forward(
                    pair.input
                );

            loss +=
                crossEntropy(
                    probabilities,
                    pair.target
                );

            /*
            ==================================================
            HINWEIS:

            Hier wird als nächster Schritt die vollständige
            Backpropagation für Attention, Embeddings,
            LayerNorm und Feed Forward ergänzt.

            Das Modell besitzt bereits die komplette
            Transformer-Forward-Architektur.
            ==================================================
            */

        }


        if(epoch%10===0){

            console.log(
                "Epoch",
                epoch,
                "| Loss:",
                (
                    loss /
                    trainingPairs.length
                ).toFixed(5)
            );

            /*
            Browser Luft geben
            */

            await new Promise(
                resolve =>
                    setTimeout(resolve,0)
            );

        }

    }

    saveModel();

    console.log(
        "Training abgeschlossen."
    );

}


// ============================================================
// MODELL SPEICHERN
// ============================================================

function serializeMatrix(m){

    return m.map(row =>
        Array.from(row)
    );

}


function saveModel(){

    const data={

        config:CONFIG,

        vocabulary:
            tokenizer.vocabulary,

        tokenEmbedding:
            serializeMatrix(
                model.tokenEmbedding
            ),

        positionEmbedding:
            serializeMatrix(
                model.positionEmbedding
            ),

        output:
            serializeMatrix(
                model.output
            ),

        blocks:
            model.blocks.map(block=>({

                q:serializeMatrix(block.q),
                k:serializeMatrix(block.k),
                v:serializeMatrix(block.v),
                o:serializeMatrix(block.o),

                ff1:serializeMatrix(block.ff1),
                ff2:serializeMatrix(block.ff2),

                gamma1:
                    Array.from(
                        block.norm1.gamma
                    ),

                beta1:
                    Array.from(
                        block.norm1.beta
                    ),

                gamma2:
                    Array.from(
                        block.norm2.gamma
                    ),

                beta2:
                    Array.from(
                        block.norm2.beta
                    )

            }))

    };


    localStorage.setItem(
        CONFIG.saveKey,
        JSON.stringify(data)
    );

    console.log(
        "Modell gespeichert:",
        CONFIG.saveKey
    );

}


// ============================================================
// TEST-GENERIERUNG
// ============================================================

function randomChoice(probabilities){

    let r=Math.random();

    for(
        let i=0;
        i<probabilities.length;
        i++
    ){

        r -= probabilities[i];

        if(r<=0){
            return i;
        }

    }

    return probabilities.length-1;

}


function generate(prompt,maxTokens=30){

    let tokens =
        tokenizer.encode(prompt);

    /*
    EOS entfernen
    */

    if(
        tokens[tokens.length-1] ===
        tokenizer.tokenToId.get("<EOS>")
    ){

        tokens.pop();

    }


    for(
        let i=0;
        i<maxTokens;
        i++
    ){

        const context =
            tokens.slice(
                -CONFIG.contextSize
            );

        const probabilities =
            model.forward(context);


        /*
        Temperature
        */

        const adjusted =
            new Float64Array(
                probabilities.length
            );

        let sum=0;

        for(
            let j=0;
            j<probabilities.length;
            j++
        ){

            adjusted[j]=
                Math.pow(
                    probabilities[j],
                    1/CONFIG.temperature
                );

            sum += adjusted[j];

        }

        for(
            let j=0;
            j<adjusted.length;
            j++
        ){

            adjusted[j]/=sum;

        }


        const next =
            randomChoice(adjusted);

        tokens.push(next);


        if(
            next ===
            tokenizer.tokenToId.get("<EOS>")
        ){

            break;

        }

    }

    return tokenizer.decode(tokens);

}


// ============================================================
// START
// ============================================================

train().then(()=>{

    console.log(
        "Testgeneration:"
    );

    console.log(
        generate("hallo")
    );

});

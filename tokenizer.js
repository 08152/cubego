"use strict";

/*
================================================================
  tokenizer.js
  Advanced BPE-like Tokenizer
  100% eigenes JavaScript
  KEINE externen Bibliotheken
================================================================
*/

const AdvancedTokenizer = (() => {

    // ==========================================================
    // KONFIGURATION
    // ==========================================================

    const CONFIG = Object.freeze({

        vocabSize: 8192,

        minFrequency: 2,

        maxInputLength: 8192,

        maxTokenLength: 64,

        maxMerges: 16000,

        byteFallback: true,

        lowercase: false,

        normalizeUnicode: true

    });


    // ==========================================================
    // SPEZIALTOKENS
    // ==========================================================

    const SPECIAL = Object.freeze({

        PAD: "<|pad|>",

        UNK: "<|unk|>",

        BOS: "<|bos|>",

        EOS: "<|eos|>",

        MASK: "<|mask|>",

        SEP: "<|sep|>",

        USER: "<|user|>",

        ASSISTANT: "<|assistant|>",

        SYSTEM: "<|system|>",

        TOOL: "<|tool|>",

        END: "<|end|>"

    });


    const SPECIAL_LIST =
        Object.values(SPECIAL);


    // ==========================================================
    // HILFSFUNKTIONEN
    // ==========================================================

    function assertString(value){

        if(
            typeof value !== "string"
        ){

            return String(
                value ?? ""
            );

        }

        return value;

    }


    function normalizeText(text){

        text =
            assertString(text);


        if(
            CONFIG.normalizeUnicode
        ){

            text =
                text.normalize("NFKC");

        }


        if(
            CONFIG.lowercase
        ){

            text =
                text.toLowerCase();

        }


        text =
            text
                .replace(/\r\n/g,"\n")
                .replace(/\r/g,"\n");


        return text;

    }


    function isWhitespace(char){

        return /\s/u.test(char);

    }


    function isLetter(char){

        return /\p{L}/u.test(char);

    }


    function isNumber(char){

        return /\p{N}/u.test(char);

    }


    function isEmoji(char){

        return /\p{Extended_Pictographic}/u.test(
            char
        );

    }


    function utf8Bytes(text){

        return new TextEncoder().encode(
            text
        );

    }


    function utf8Decode(bytes){

        return new TextDecoder(
            "utf-8",
            {
                fatal:false
            }
        ).decode(bytes);

    }


    // ==========================================================
    // PRIORITY QUEUE
    // ==========================================================

    class PriorityQueue {

        constructor(){

            this.items=[];

        }


        push(item){

            this.items.push(item);

        }


        popBest(){

            if(
                this.items.length===0
            ){

                return null;

            }


            let bestIndex=0;


            for(
                let i=1;
                i<this.items.length;
                i++
            ){

                if(
                    this.items[i].score >
                    this.items[bestIndex].score
                ){

                    bestIndex=i;

                }

            }


            return this.items.splice(
                bestIndex,
                1
            )[0];

        }


        get size(){

            return this.items.length;

        }

    }


    // ==========================================================
    // TOKEN OBJECT
    // ==========================================================

    class Token {

        constructor(
            id,
            text,
            type="subword"
        ){

            this.id=id;

            this.text=text;

            this.type=type;

        }

    }


    // ==========================================================
    // TOKENIZER
    // ==========================================================

    class Engine {

        constructor(){

            this.vocabulary=[];

            this.tokenToId=
                new Map();

            this.idToToken=
                new Map();

            this.tokenType=
                new Map();

            this.merges=[];

            this.mergeRanks=
                new Map();

            this.frequencies=
                new Map();

            this.initialized=false;

            this.initializeSpecialTokens();

        }


        // ======================================================
        // SPEZIALTOKENS INITIALISIEREN
        // ======================================================

        initializeSpecialTokens(){

            for(
                const token
                of SPECIAL_LIST
            ){

                this.addToken(
                    token,
                    "special"
                );

            }

        }


        // ======================================================
        // TOKEN HINZUFÜGEN
        // ======================================================

        addToken(
            token,
            type="subword"
        ){

            if(
                this.tokenToId.has(token)
            ){

                return this.tokenToId.get(
                    token
                );

            }


            const id =
                this.vocabulary.length;


            this.vocabulary.push(token);

            this.tokenToId.set(
                token,
                id
            );

            this.idToToken.set(
                id,
                token
            );

            this.tokenType.set(
                id,
                type
            );


            return id;

        }


        // ======================================================
        // BASIS-TOKENISIERUNG
        // ======================================================

        preTokenize(text){

            text =
                normalizeText(text);


            const result=[];

            let current="";


            const flush=()=>{

                if(current){

                    result.push(
                        current
                    );

                    current="";

                }

            };


            for(
                const char
                of text
            ){

                /*
                Whitespace
                */

                if(
                    isWhitespace(char)
                ){

                    flush();

                    /*
                    Leerzeichen werden
                    als eigener Tokenbereich
                    behalten.
                    */

                    result.push(
                        "▁"
                    );

                    continue;

                }


                /*
                Emojis
                */

                if(
                    isEmoji(char)
                ){

                    flush();

                    result.push(
                        char
                    );

                    continue;

                }


                /*
                Zahlen
                */

                if(
                    isNumber(char)
                ){

                    if(
                        current &&
                        !isNumber(
                            current.at(-1)
                        )
                    ){

                        flush();

                    }

                    current+=char;

                    continue;

                }


                /*
                Buchstaben
                */

                if(
                    isLetter(char)
                ){

                    if(
                        current &&
                        !isLetter(
                            current.at(-1)
                        )
                    ){

                        flush();

                    }

                    current+=char;

                    continue;

                }


                /*
                Satzzeichen / Symbole
                */

                flush();

                result.push(
                    char
                );

            }


            flush();


            return result;

        }


        // ======================================================
        // ZEICHEN → BASIS-SYMBOLE
        // ======================================================

        wordToSymbols(word){

            const chars =
                Array.from(word);

            return chars.map(
                char => char
            );

        }


        // ======================================================
        // FREQUENZEN SAMMELN
        // ======================================================

        collectCorpus(texts){

            const wordSequences=[];


            for(
                const text
                of texts
            ){

                const words =
                    this.preTokenize(
                        text
                    );

                wordSequences.push(
                    words
                );


                for(
                    const word
                    of words
                ){

                    this.frequencies.set(
                        word,
                        (
                            this.frequencies.get(
                                word
                            ) || 0
                        ) + 1
                    );

                }

            }


            return wordSequences;

        }


        // ======================================================
        // PAAR-FREQUENZEN
        // ======================================================

        countPairs(wordSequences){

            const pairs=
                new Map();


            for(
                const words
                of wordSequences
            ){

                for(
                    const word
                    of words
                ){

                    const symbols =
                        this.wordToSymbols(
                            word
                        );


                    for(
                        let i=0;
                        i<symbols.length-1;
                        i++
                    ){

                        const key =
                            symbols[i] +
                            "\u0001" +
                            symbols[i+1];


                        pairs.set(
                            key,
                            (
                                pairs.get(key)
                                || 0
                            ) + 1
                        );

                    }

                }

            }


            return pairs;

        }


        // ======================================================
        // BESTES PAAR
        // ======================================================

        findBestPair(pairs){

            let best=null;

            let bestCount=0;


            for(
                const [
                    key,
                    count
                ]
                of pairs
            ){

                if(
                    count > bestCount
                ){

                    bestCount=count;

                    const parts =
                        key.split(
                            "\u0001"
                        );


                    best={
                        a:parts[0],
                        b:parts[1],
                        count
                    };

                }

            }


            return best;

        }


        // ======================================================
        // PAAR MERGEN
        // ======================================================

        mergeSymbols(
            symbols,
            a,
            b
        ){

            const output=[];

            let i=0;


            while(
                i<symbols.length
            ){

                if(
                    i<
                    symbols.length-1 &&
                    symbols[i]===a &&
                    symbols[i+1]===b
                ){

                    output.push(
                        a+b
                    );

                    i+=2;

                }

                else{

                    output.push(
                        symbols[i]
                    );

                    i++;

                }

            }


            return output;

        }


        // ======================================================
        // TRAINING DER BPE-MERGES
        // ======================================================

        trainBPE(texts){

            console.log(
                "BPE-Training gestartet..."
            );


            let sequences =
                this.collectCorpus(
                    texts
                );


            /*
            Häufige Einzelzeichen
            */

            const characterFrequency =
                new Map();


            for(
                const words
                of sequences
            ){

                for(
                    const word
                    of words
                ){

                    for(
                        const char
                        of this.wordToSymbols(
                            word
                        )
                    ){

                        characterFrequency.set(
                            char,
                            (
                                characterFrequency.get(
                                    char
                                ) || 0
                            ) + 1
                        );

                    }

                }

            }


            const characters =
                [...characterFrequency.entries()]
                .sort(
                    (a,b)=>b[1]-a[1]
                );


            for(
                const [
                    char
                ]
                of characters
            ){

                if(
                    this.vocabulary.length >=
                    CONFIG.vocabSize
                ){

                    break;

                }


                this.addToken(
                    char,
                    "character"
                );

            }


            /*
            BPE MERGES
            */

            for(
                let iteration=0;
                iteration<
                    CONFIG.maxMerges;
                iteration++
            ){

                if(
                    this.vocabulary.length >=
                    CONFIG.vocabSize
                ){

                    break;

                }


                const pairs =
                    this.countPairs(
                        sequences
                    );


                const best =
                    this.findBestPair(
                        pairs
                    );


                if(
                    !best ||
                    best.count <
                    CONFIG.minFrequency
                ){

                    break;

                }


                const merged =
                    best.a +
                    best.b;


                if(
                    merged.length >
                    CONFIG.maxTokenLength
                ){

                    break;

                }


                if(
                    !this.tokenToId.has(
                        merged
                    )
                ){

                    this.addToken(
                        merged,
                        "subword"
                    );

                }


                this.merges.push([
                    best.a,
                    best.b
                ]);


                this.mergeRanks.set(
                    best.a +
                    "\u0001" +
                    best.b,
                    iteration
                );


                /*
                Merge in allen Wörtern
                */

                sequences =
                    sequences.map(
                        words =>
                            words.map(
                                word =>
                                    this.mergeWord(
                                        word,
                                        best.a,
                                        best.b
                                    )
                            )
                    );


                if(
                    iteration % 100 === 0
                ){

                    console.log(
                        "Merge",
                        iteration,
                        "| Vocab:",
                        this.vocabulary.length,
                        "| Paar:",
                        best.a,
                        "+",
                        best.b,
                        "| Häufigkeit:",
                        best.count
                    );

                }

            }


            this.initialized=true;


            console.log(
                "BPE fertig.",
                "Vocabulary:",
                this.vocabulary.length,
                "Merges:",
                this.merges.length
            );


            return this.export();

        }


        // ======================================================
        // EIN WORT MERGEN
        // ======================================================

        mergeWord(
            word,
            a,
            b
        ){

            let symbols =
                this.wordToSymbols(
                    word
                );


            let changed=true;


            while(changed){

                changed=false;


                for(
                    let i=0;
                    i<symbols.length-1;
                    i++
                ){

                    if(
                        symbols[i]===a &&
                        symbols[i+1]===b
                    ){

                        symbols =
                            this.mergeSymbols(
                                symbols,
                                a,
                                b
                            );

                        changed=true;

                        break;

                    }

                }

            }


            return symbols.join("");

        }


        // ======================================================
        // TOKENISIERUNG MIT BPE
        // ======================================================

        encodeWord(word){

            let symbols =
                this.wordToSymbols(
                    word
                );


            if(
                symbols.length===0
            ){

                return [];

            }


            /*
            BPE-Merges nach Rang anwenden.
            */

            while(
                symbols.length>1
            ){

                let bestIndex=-1;

                let bestRank=Infinity;


                for(
                    let i=0;
                    i<symbols.length-1;
                    i++
                ){

                    const key =
                        symbols[i] +
                        "\u0001" +
                        symbols[i+1];


                    const rank =
                        this.mergeRanks.get(
                            key
                        );


                    if(
                        rank !== undefined &&
                        rank < bestRank
                    ){

                        bestRank=rank;

                        bestIndex=i;

                    }

                }


                if(bestIndex===-1){

                    break;

                }


                symbols.splice(
                    bestIndex,
                    2,
                    symbols[
                        bestIndex
                    ] +
                    symbols[
                        bestIndex+1
                    ]
                );

            }


            return symbols;

        }


        // ======================================================
        // TEXT → TOKEN STRUKTUR
        // ======================================================

        encode(text,options={}){

            const addBOS =
                options.addBOS !== false;

            const addEOS =
                options.addEOS !== false;


            text =
                normalizeText(text);


            /*
            Spezialtokens erkennen
            */

            const pieces =
                this.preTokenize(
                    text
                );


            const ids=[];


            if(addBOS){

                ids.push(
                    this.tokenToId.get(
                        SPECIAL.BOS
                    )
                );

            }


            for(
                const piece
                of pieces
            ){

                /*
                Direkt vorhandener Token
                */

                if(
                    this.tokenToId.has(
                        piece
                    )
                ){

                    ids.push(
                        this.tokenToId.get(
                            piece
                        )
                    );

                    continue;

                }


                /*
                BPE
                */

                const subwords =
                    this.encodeWord(
                        piece
                    );


                for(
                    const subword
                    of subwords
                ){

                    if(
                        this.tokenToId.has(
                            subword
                        )
                    ){

                        ids.push(
                            this.tokenToId.get(
                                subword
                            )
                        );

                    }

                    else if(
                        CONFIG.byteFallback
                    ){

                        /*
                        Byte-Fallback
                        */

                        const bytes =
                            utf8Bytes(
                                subword
                            );


                        for(
                            const byte
                            of bytes
                        ){

                            const byteToken =
                                "<|byte:" +
                                byte +
                                "|>";


                            let id =
                                this.tokenToId.get(
                                    byteToken
                                );


                            if(
                                id===undefined
                            ){

                                id=this.addToken(
                                    byteToken,
                                    "byte"
                                );

                            }


                            ids.push(id);

                        }

                    }

                    else{

                        ids.push(
                            this.tokenToId.get(
                                SPECIAL.UNK
                            )
                        );

                    }

                }

            }


            if(addEOS){

                ids.push(
                    this.tokenToId.get(
                        SPECIAL.EOS
                    )
                );

            }


            return ids;

        }


        // ======================================================
        // IDS → TEXT
        // ======================================================

        decode(ids){

            let output="";

            let byteBuffer=[];


            const flushBytes=()=>{

                if(
                    byteBuffer.length
                ){

                    output +=
                        utf8Decode(
                            new Uint8Array(
                                byteBuffer
                            )
                        );

                    byteBuffer=[];

                }

            };


            for(
                const id
                of ids
            ){

                const token =
                    this.idToToken.get(
                        id
                    );


                if(
                    token===undefined
                ){

                    continue;

                }


                /*
                Spezialtokens
                */

                if(
                    SPECIAL_LIST.includes(
                        token
                    )
                ){

                    flushBytes();

                    continue;

                }


                /*
                Byte-Fallback
                */

                if(
                    token.startsWith(
                        "<|byte:"
                    )
                ){

                    const number =
                        Number(
                            token.slice(
                                7,
                                -2
                            )
                        );


                    if(
                        Number.isFinite(
                            number
                        )
                    ){

                        byteBuffer.push(
                            number
                        );

                    }

                    continue;

                }


                flushBytes();

                output += token;

            }


            flushBytes();


            /*
            BPE-Leerzeichen wiederherstellen
            */

            output =
                output.replace(
                    /▁/g,
                    " "
                );


            return output.trim();

        }


        // ======================================================
        // BATCH ENCODE
        // ======================================================

        encodeBatch(
            texts,
            options={}
        ){

            return texts.map(
                text =>
                    this.encode(
                        text,
                        options
                    )
            );

        }


        // ======================================================
        // BATCH DECODE
        // ======================================================

        decodeBatch(idsList){

            return idsList.map(
                ids =>
                    this.decode(ids)
            );

        }


        // ======================================================
        // TOKEN INFORMATION
        // ======================================================

        getToken(id){

            return {

                id,

                text:
                    this.idToToken.get(
                        id
                    ),

                type:
                    this.tokenType.get(
                        id
                    ) || "unknown"

            };

        }


        // ======================================================
        // STATISTIK
        // ======================================================

        statistics(){

            return {

                vocabularySize:
                    this.vocabulary.length,

                merges:
                    this.merges.length,

                frequencies:
                    this.frequencies.size,

                specialTokens:
                    SPECIAL_LIST.length

            };

        }


        // ======================================================
        // EXPORT
        // ======================================================

        export(){

            return {

                version:1,

                config:{
                    ...CONFIG
                },

                special:{
                    ...SPECIAL
                },

                vocabulary:[
                    ...this.vocabulary
                ],

                merges:
                    this.merges.map(
                        pair => [...pair]
                    ),

                mergeRanks:
                    Object.fromEntries(
                        this.mergeRanks
                    ),

                frequencies:
                    Object.fromEntries(
                        this.frequencies
                    )

            };

        }


        // ======================================================
        // IMPORT
        // ======================================================

        import(data){

            if(
                !data ||
                !Array.isArray(
                    data.vocabulary
                )
            ){

                throw new Error(
                    "Ungültiges Tokenizer-Modell."
                );

            }


            this.vocabulary =
                [...data.vocabulary];


            this.merges =
                (data.merges || [])
                .map(
                    pair => [...pair]
                );


            this.mergeRanks =
                new Map(
                    Object.entries(
                        data.mergeRanks || {}
                    ).map(
                        ([key,value]) =>
                            [key,Number(value)]
                    )
                );


            this.frequencies =
                new Map(
                    Object.entries(
                        data.frequencies || {}
                    ).map(
                        ([key,value]) =>
                            [key,Number(value)]
                    )
                );


            this.rebuildMaps();

            this.initialized=true;

        }


        // ======================================================
        // MAPS NEU AUFBAUEN
        // ======================================================

        rebuildMaps(){

            this.tokenToId.clear();
            this.idToToken.clear();
            this.tokenType.clear();


            for(
                let i=0;
                i<this.vocabulary.length;
                i++
            ){

                const token =
                    this.vocabulary[i];


                this.tokenToId.set(
                    token,
                    i
                );


                this.idToToken.set(
                    i,
                    token
                );


                if(
                    SPECIAL_LIST.includes(
                        token
                    )
                ){

                    this.tokenType.set(
                        i,
                        "special"
                    );

                }

                else if(
                    token.startsWith(
                        "<|byte:"
                    )
                ){

                    this.tokenType.set(
                        i,
                        "byte"
                    );

                }

                else if(
                    token.length===1
                ){

                    this.tokenType.set(
                        i,
                        "character"
                    );

                }

                else{

                    this.tokenType.set(
                        i,
                        "subword"
                    );

                }

            }

        }


        // ======================================================
        // TOKEN-ID TEST
        // ======================================================

        hasToken(token){

            return this.tokenToId.has(
                token
            );

        }


        // ======================================================
        // TEXT → TOKENS
        // ======================================================

        tokens(text){

            return this.encode(
                text,
                {
                    addBOS:false,
                    addEOS:false
                }
            ).map(
                id =>
                    this.getToken(id)
            );

        }


        // ======================================================
        // SPEICHERGRÖSSE
        // ======================================================

        approximateSize(){

            return JSON.stringify(
                this.export()
            ).length;

        }

    }


    // ==========================================================
    // PUBLIC API
    // ==========================================================

    return {

        create(){

            return new Engine();

        },

        SPECIAL,

        SPECIAL_LIST,

        CONFIG

    };

})();


// ============================================================
// BROWSER
// ============================================================

if(
    typeof window !== "undefined"
){

    window.AdvancedTokenizer =
        AdvancedTokenizer;

}


// ============================================================
// NODE
// ============================================================

if(
    typeof module !== "undefined" &&
    module.exports
){

    module.exports =
        AdvancedTokenizer;

}

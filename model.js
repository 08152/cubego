"use strict";

/*
===========================================================
 LUMORA PRO / LARGE
 Hybrid Language Model

 Hauptziel:
 - JSON-Daten aus DATEN/ zuverlässig verarbeiten
 - konkrete Q&A-Daten stark lernen
 - neuronalen Fallback bereitstellen
 - stabil speichern/laden
===========================================================
*/

const fs = require("fs");
const path = require("path");

/* =========================================================
   DEFAULT CONFIG
========================================================= */

const DEFAULT_CONFIG = {
  modelType: "LUMORA-PRO",

  vocabSize: 8192,

  contextSize: 256,

  embeddingSize: 128,

  layers: 4,

  heads: 4,

  headSize: 32,

  feedForwardSize: 512,

  maxNgramOrder: 8,

  maxNgramEntries: 100000,

  learningRate: 0.00025,

  minLearningRate: 0.00002,

  temperature: 0.75,

  topK: 40,

  topP: 0.92,

  repetitionPenalty: 1.08,

  seed: 1337
};


/* =========================================================
   HELPER
========================================================= */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}


function randomSeeded(seed) {
  let s = seed >>> 0;

  return function () {
    s += 0x6D2B79F5;

    let t = s;

    t = Math.imul(
      t ^ (t >>> 15),
      t | 1
    );

    t ^= t +
      Math.imul(
        t ^ (t >>> 7),
        t | 61
      );

    return (
      (
        (t ^ (t >>> 14)) >>> 0
      ) / 4294967296
    );
  };
}


function randNormal(rng) {
  let u = 0;
  let v = 0;

  while (u === 0) {
    u = rng();
  }

  while (v === 0) {
    v = rng();
  }

  return Math.sqrt(
    -2 * Math.log(u)
  ) *
  Math.cos(
    2 * Math.PI * v
  );
}


function softmax(logits, temperature = 1) {

  temperature =
    Math.max(
      0.05,
      Number(temperature) || 1
    );

  let max = -Infinity;

  for (
    let i = 0;
    i < logits.length;
    i++
  ) {

    const value =
      logits[i] /
      temperature;

    if (value > max) {
      max = value;
    }
  }

  const probabilities =
    new Float32Array(
      logits.length
    );

  let sum = 0;

  for (
    let i = 0;
    i < logits.length;
    i++
  ) {

    const value =
      Math.exp(
        clamp(
          logits[i] /
            temperature -
            max,
          -80,
          0
        )
      );

    probabilities[i] = value;
    sum += value;
  }

  if (
    !Number.isFinite(sum) ||
    sum <= 0
  ) {

    const value =
      1 /
      logits.length;

    probabilities.fill(value);

    return probabilities;
  }

  for (
    let i = 0;
    i < probabilities.length;
    i++
  ) {
    probabilities[i] /=
      sum;
  }

  return probabilities;
}


function gelu(x) {

  const c =
    Math.sqrt(
      2 / Math.PI
    );

  return (
    0.5 *
    x *
    (
      1 +
      Math.tanh(
        c *
        (
          x +
          0.044715 *
          x *
          x *
          x
        )
      )
    )
  );
}


/* =========================================================
   PARAMETER
========================================================= */

class Parameter {

  constructor(
    size,
    rng,
    std = 0.02
  ) {

    this.data =
      new Float32Array(size);

    this.grad =
      new Float32Array(size);

    this.m =
      new Float32Array(size);

    this.v =
      new Float32Array(size);

    for (
      let i = 0;
      i < size;
      i++
    ) {

      this.data[i] =
        randNormal(rng) * std;
    }
  }

  zeroGrad() {
    this.grad.fill(0);
  }
}


/* =========================================================
   LAYER NORMALIZATION
========================================================= */

class LayerNorm {

  constructor(
    size,
    rng
  ) {

    this.size = size;

    this.gamma =
      new Parameter(
        size,
        rng,
        0
      );

    this.beta =
      new Parameter(
        size,
        rng,
        0
      );

    this.gamma.data.fill(1);
    this.beta.data.fill(0);
  }

  forward(input) {

    let mean = 0;

    for (
      let i = 0;
      i < this.size;
      i++
    ) {
      mean += input[i];
    }

    mean /=
      this.size;

    let variance = 0;

    for (
      let i = 0;
      i < this.size;
      i++
    ) {

      const d =
        input[i] -
        mean;

      variance +=
        d * d;
    }

    variance /=
      this.size;

    const invStd =
      1 /
      Math.sqrt(
        variance + 1e-5
      );

    const result =
      new Float32Array(
        this.size
      );

    for (
      let i = 0;
      i < this.size;
      i++
    ) {

      result[i] =
        (
          (
            input[i] -
            mean
          ) *
          invStd
        ) *
        this.gamma.data[i]
        +
        this.beta.data[i];
    }

    return result;
  }
}


/* =========================================================
   TRANSFORMER BLOCK
========================================================= */

class TransformerBlock {

  constructor(
    config,
    rng
  ) {

    this.config =
      config;

    const d =
      config.embeddingSize;

    const attentionSize =
      config.heads *
      config.headSize;

    const ff =
      config.feedForwardSize;

    this.q =
      new Parameter(
        d * attentionSize,
        rng,
        0.02
      );

    this.k =
      new Parameter(
        d * attentionSize,
        rng,
        0.02
      );

    this.v =
      new Parameter(
        d * attentionSize,
        rng,
        0.02
      );

    this.o =
      new Parameter(
        attentionSize * d,
        rng,
        0.02
      );

    this.ff1 =
      new Parameter(
        d * ff,
        rng,
        0.02
      );

    this.ff2 =
      new Parameter(
        ff * d,
        rng,
        0.02
      );

    this.ffBias1 =
      new Parameter(
        ff,
        rng,
        0
      );

    this.ffBias2 =
      new Parameter(
        d,
        rng,
        0
      );

    this.norm1 =
      new LayerNorm(
        d,
        rng
      );

    this.norm2 =
      new LayerNorm(
        d,
        rng
      );
  }


  linear(
    matrix,
    bias,
    input,
    outputSize,
    inputSize
  ) {

    const result =
      new Float32Array(
        outputSize
      );

    for (
      let row = 0;
      row < outputSize;
      row++
    ) {

      let sum =
        bias
          ? bias[row]
          : 0;

      const start =
        row *
        inputSize;

      for (
        let col = 0;
        col < inputSize;
        col++
      ) {

        sum +=
          matrix[
            start + col
          ] *
          input[col];
      }

      result[row] =
        sum;
    }

    return result;
  }


  forward(sequence) {

    const cfg =
      this.config;

    const d =
      cfg.embeddingSize;

    const heads =
      cfg.heads;

    const headSize =
      cfg.headSize;

    const attentionSize =
      heads *
      headSize;

    const length =
      sequence.length;

    if (length === 0) {
      return [];
    }


    /* -------------------------
       PRE-NORM
    ------------------------- */

    const normalized =
      new Array(length);

    for (
      let i = 0;
      i < length;
      i++
    ) {

      normalized[i] =
        this.norm1.forward(
          sequence[i]
        );
    }


    /* -------------------------
       Q K V
    ------------------------- */

    const Q =
      new Array(length);

    const K =
      new Array(length);

    const V =
      new Array(length);

    for (
      let t = 0;
      t < length;
      t++
    ) {

      Q[t] =
        this.linear(
          this.q.data,
          null,
          normalized[t],
          attentionSize,
          d
        );

      K[t] =
        this.linear(
          this.k.data,
          null,
          normalized[t],
          attentionSize,
          d
        );

      V[t] =
        this.linear(
          this.v.data,
          null,
          normalized[t],
          attentionSize,
          d
        );
    }


    /* -------------------------
       CAUSAL ATTENTION
    ------------------------- */

    const attentionResult =
      new Array(length);

    const scale =
      1 /
      Math.sqrt(headSize);

    for (
      let t = 0;
      t < length;
      t++
    ) {

      const combined =
        new Float32Array(
          attentionSize
        );

      for (
        let head = 0;
        head < heads;
        head++
      ) {

        const offset =
          head *
          headSize;

        const scores =
          new Float32Array(
            t + 1
          );

        let maxScore =
          -Infinity;

        for (
          let j = 0;
          j <= t;
          j++
        ) {

          let score = 0;

          for (
            let x = 0;
            x < headSize;
            x++
          ) {

            score +=
              Q[t][
                offset + x
              ] *
              K[j][
                offset + x
              ];
          }

          score *= scale;

          scores[j] =
            score;

          if (
            score >
            maxScore
          ) {
            maxScore =
              score;
          }
        }

        let sum = 0;

        for (
          let j = 0;
          j <= t;
          j++
        ) {

          scores[j] =
            Math.exp(
              clamp(
                scores[j] -
                  maxScore,
                -60,
                0
              )
            );

          sum +=
            scores[j];
        }

        if (sum <= 0) {
          sum = 1;
        }

        for (
          let j = 0;
          j <= t;
          j++
        ) {

          const weight =
            scores[j] /
            sum;

          for (
            let x = 0;
            x < headSize;
            x++
          ) {

            combined[
              offset + x
            ] +=
              weight *
              V[j][
                offset + x
              ];
          }
        }
      }


      attentionResult[t] =
        this.linear(
          this.o.data,
          null,
          combined,
          d,
          attentionSize
        );
    }


    /* -------------------------
       RESIDUAL 1
    ------------------------- */

    const residual =
      new Array(length);

    for (
      let t = 0;
      t < length;
      t++
    ) {

      const out =
        new Float32Array(d);

      for (
        let i = 0;
        i < d;
        i++
      ) {

        out[i] =
          sequence[t][i] +
          attentionResult[t][i];
      }

      residual[t] =
        out;
    }


    /* -------------------------
       FEED FORWARD
    ------------------------- */

    const output =
      new Array(length);

    for (
      let t = 0;
      t < length;
      t++
    ) {

      const normalized2 =
        this.norm2.forward(
          residual[t]
        );


      const hidden =
        this.linear(
          this.ff1.data,
          this.ffBias1.data,
          normalized2,
          cfg.feedForwardSize,
          d
        );


      for (
        let i = 0;
        i < hidden.length;
        i++
      ) {

        hidden[i] =
          gelu(
            hidden[i]
          );
      }


      const ffOutput =
        this.linear(
          this.ff2.data,
          this.ffBias2.data,
          hidden,
          d,
          cfg.feedForwardSize
        );


      const final =
        new Float32Array(d);

      for (
        let i = 0;
        i < d;
        i++
      ) {

        final[i] =
          residual[t][i] +
          ffOutput[i];
      }

      output[t] =
        final;
    }

    return output;
  }
}


/* =========================================================
   LUMORA
========================================================= */

class LanguageModel {

  constructor(
    config = {}
  ) {

    this.config = {
      ...DEFAULT_CONFIG,
      ...config
    };


    /* -------------------------
       Konsistenz
    ------------------------- */

    this.config.headSize =
      Math.max(
        1,
        Number(
          this.config.headSize
        ) || 32
      );

    this.config.heads =
      Math.max(
        1,
        Number(
          this.config.heads
        ) || 4
      );

    this.config.embeddingSize =
      this.config.heads *
      this.config.headSize;


    this.rng =
      randomSeeded(
        this.config.seed
      );


    const vocabSize =
      this.config.vocabSize;

    const d =
      this.config.embeddingSize;


    /* -------------------------
       Embeddings
    ------------------------- */

    this.tokenEmbedding =
      new Parameter(
        vocabSize * d,
        this.rng,
        0.025
      );


    this.positionEmbedding =
      new Parameter(
        this.config.contextSize * d,
        this.rng,
        0.01
      );


    this.outputBias =
      new Parameter(
        vocabSize,
        this.rng,
        0
      );


    /* -------------------------
       Transformer
    ------------------------- */

    this.blocks = [];

    for (
      let i = 0;
      i < this.config.layers;
      i++
    ) {

      this.blocks.push(
        new TransformerBlock(
          this.config,
          this.rng
        )
      );
    }


    /* =====================================================
       WICHTIG:
       TRAININGSMEMORY

       key:
         Kontext-Token-IDs

       value:
         nächste Token + Anzahl
    ===================================================== */

    this.ngrams =
      new Map();


    /* =====================================================
       Gelerntes Q&A
    ===================================================== */

    this.trainingExamples =
      [];


    this.trainingStep =
      0;

    this.totalTokensSeen =
      0;

    this.lastLoss =
      null;

    this.loadedTrainingFiles =
      0;

    this.loadedTrainingExamples =
      0;
  }


  /* =======================================================
     DATEN LADEN
  ======================================================= */

  static findJSONFiles(
    dataDirectory
  ) {

    const result = [];

    if (
      !fs.existsSync(
        dataDirectory
      )
    ) {
      return result;
    }


    function scan(directory) {

      let entries;

      try {

        entries =
          fs.readdirSync(
            directory,
            {
              withFileTypes: true
            }
          );

      } catch {
        return;
      }


      for (
        const entry of entries
      ) {

        const fullPath =
          path.join(
            directory,
            entry.name
          );


        if (
          entry.isDirectory()
        ) {

          scan(
            fullPath
          );

          continue;
        }


        if (
          entry.isFile() &&
          entry.name
            .toLowerCase()
            .endsWith(".json")
        ) {

          result.push(
            fullPath
          );
        }
      }
    }


    scan(
      dataDirectory
    );


    result.sort(
      (a, b) =>
        a.localeCompare(
          b,
          "de",
          {
            numeric: true,
            sensitivity: "base"
          }
        )
    );


    return result;
  }


  static extractExamples(
    value,
    result = []
  ) {

    if (
      value === null ||
      value === undefined
    ) {
      return result;
    }


    if (
      Array.isArray(value)
    ) {

      for (
        const item of value
      ) {

        this.extractExamples(
          item,
          result
        );
      }

      return result;
    }


    if (
      typeof value !==
      "object"
    ) {
      return result;
    }


    const questionKeys = [
      "frage",
      "question",
      "prompt",
      "input",
      "user"
    ];

    const answerKeys = [
      "antwort",
      "answer",
      "response",
      "output",
      "assistant"
    ];


    let question = null;
    let answer = null;


    for (
      const key of questionKeys
    ) {

      if (
        typeof value[key] ===
        "string"
      ) {

        question =
          value[key];

        break;
      }
    }


    for (
      const key of answerKeys
    ) {

      if (
        typeof value[key] ===
        "string"
      ) {

        answer =
          value[key];

        break;
      }
    }


    if (
      question !== null &&
      answer !== null
    ) {

      const q =
        question
          .trim();

      const a =
        answer
          .trim();

      if (
        q.length > 0 &&
        a.length > 0
      ) {

        result.push({
          question: q,
          answer: a
        });
      }
    }


    for (
      const key of Object.keys(
        value
      )
    ) {

      const child =
        value[key];

      if (
        child &&
        typeof child ===
        "object"
      ) {

        this.extractExamples(
          child,
          result
        );
      }
    }


    return result;
  }


  static loadTrainingData(
    dataDirectory
  ) {

    const files =
      this.findJSONFiles(
        dataDirectory
      );

    const examples =
      [];

    const seen =
      new Set();

    let validFiles =
      0;


    for (
      const file of files
    ) {

      let parsed;

      try {

        const raw =
          fs.readFileSync(
            file,
            "utf8"
          );

        parsed =
          JSON.parse(
            raw
          );

        validFiles++;

      } catch (error) {

        continue;
      }


      const extracted =
        this.extractExamples(
          parsed,
          []
        );


      for (
        const example of extracted
      ) {

        const normalizedQuestion =
          example.question
            .replace(/\s+/g, " ")
            .trim()
            .toLowerCase();

        const normalizedAnswer =
          example.answer
            .replace(/\s+/g, " ")
            .trim();


        const key =
          normalizedQuestion +
          "\n" +
          normalizedAnswer;


        if (
          seen.has(key)
        ) {
          continue;
        }


        seen.add(key);

        examples.push(
          example
        );
      }
    }


    return {
      files,
      validFiles,
      examples
    };
  }


  loadTrainingData(
    dataDirectory
  ) {

    const loaded =
      LanguageModel.loadTrainingData(
        dataDirectory
      );


    this.loadedTrainingFiles =
      loaded.validFiles;

    this.loadedTrainingExamples =
      loaded.examples.length;


    return loaded;
  }


  /* =======================================================
     TOKENIZER HILFSFUNKTION
  ======================================================= */

  encodeText(
    tokenizer,
    text
  ) {

    if (
      !tokenizer ||
      typeof tokenizer.encode !==
      "function"
    ) {
      return [];
    }


    try {

      const result =
        tokenizer.encode(
          String(text)
        );


      if (
        Array.isArray(result)
      ) {
        return result
          .map(Number)
          .filter(
            id =>
              Number.isInteger(id)
          );
      }


      if (
        result &&
        Array.isArray(
          result.ids
        )
      ) {

        return result.ids
          .map(Number)
          .filter(
            id =>
              Number.isInteger(id)
          );
      }

    } catch {
      return [];
    }


    return [];
  }


  /* =======================================================
     TRAININGSDATEN DIREKT VERARBEITEN
  ======================================================= */

  trainFromDirectory(
    dataDirectory,
    tokenizer,
    options = {}
  ) {

    const loaded =
      this.loadTrainingData(
        dataDirectory
      );


    if (
      loaded.examples.length === 0
    ) {

      return {
        files:
          loaded.files.length,

        examples:
          0,

        trained:
          0,

        error:
          "Keine gültigen Frage/Antwort-Daten in DATEN gefunden."
      };
    }


    let trained =
      0;


    for (
      const example of
      loaded.examples
    ) {

      const userText =
        "<|user|>\n" +
        example.question +
        "\n<|assistant|>\n";


      const assistantText =
        example.answer +
        "\n<|end|>";


      const questionTokens =
        this.encodeText(
          tokenizer,
          userText
        );


      const answerTokens =
        this.encodeText(
          tokenizer,
          assistantText
        );


      if (
        questionTokens.length === 0 ||
        answerTokens.length === 0
      ) {
        continue;
      }


      const fullSequence =
        questionTokens.concat(
          answerTokens
        );


      this.rememberSequence(
        fullSequence
      );


      this.trainingExamples.push({
        question:
          questionTokens,

        answer:
          answerTokens
      });


      trained++;
    }


    this.limitTrainingExamples();


    return {
      files:
        loaded.files.length,

      validFiles:
        loaded.validFiles,

      examples:
        loaded.examples.length,

      trained,

      ngramEntries:
        this.ngrams.size
    };
  }


  /* =======================================================
     N-GRAM TRAINING
  ======================================================= */

  rememberSequence(
    sequence
  ) {

    if (
      !Array.isArray(sequence)
    ) {
      return;
    }


    const clean =
      sequence
        .map(Number)
        .filter(
          id =>
            Number.isInteger(id) &&
            id >= 0 &&
            id <
              this.config.vocabSize
        );


    if (
      clean.length < 2
    ) {
      return;
    }


    const maxOrder =
      Math.min(
        this.config.maxNgramOrder,
        clean.length - 1
      );


    for (
      let order = 1;
      order <= maxOrder;
      order++
    ) {

      for (
        let i = 0;
        i + order < clean.length;
        i++
      ) {

        const context =
          clean.slice(
            i,
            i + order
          );


        const next =
          clean[
            i + order
          ];


        const key =
          context.join(",");


        let table =
          this.ngrams.get(
            key
          );


        if (!table) {

          table =
            new Map();

          this.ngrams.set(
            key,
            table
          );
        }


        table.set(
          next,
          (
            table.get(next) ||
            0
          ) + 1
        );
      }
    }


    this.limitNgrams();
  }


  limitNgrams() {

    const maxEntries =
      Number(
        this.config.maxNgramEntries
      );


    if (
      this.ngrams.size <=
      maxEntries
    ) {
      return;
    }


    const entries =
      Array.from(
        this.ngrams.entries()
      );


    entries.sort(
      (a, b) => {

        let ca = 0;
        let cb = 0;


        for (
          const value of
          a[1].values()
        ) {
          ca += value;
        }


        for (
          const value of
          b[1].values()
        ) {
          cb += value;
        }


        return cb - ca;
      }
    );


    this.ngrams =
      new Map(
        entries.slice(
          0,
          maxEntries
        )
      );
  }


  limitTrainingExamples() {

    const maximum =
      20000;


    if (
      this.trainingExamples.length >
      maximum
    ) {

      this.trainingExamples =
        this.trainingExamples.slice(
          -maximum
        );
    }
  }


  /* =======================================================
     N-GRAM PREDICTION
  ======================================================= */

  getNgramCandidates(
    context
  ) {

    if (
      !Array.isArray(context) ||
      context.length === 0
    ) {
      return null;
    }


    const maxOrder =
      Math.min(
        this.config.maxNgramOrder,
        context.length
      );


    for (
      let order = maxOrder;
      order >= 1;
      order--
    ) {

      const recent =
        context.slice(
          -order
        );


      const key =
        recent.join(",");


      const table =
        this.ngrams.get(
          key
        );


      if (
        table &&
        table.size > 0
      ) {

        return {
          order,
          table
        };
      }
    }


    return null;
  }


  ngramPrediction(
    context
  ) {

    const found =
      this.getNgramCandidates(
        context
      );


    if (!found) {
      return null;
    }


    let bestToken =
      null;

    let bestCount =
      -1;

    let total =
      0;


    for (
      const [
        token,
        count
      ]
      of found.table
    ) {

      total += count;


      if (
        count >
        bestCount
      ) {

        bestCount =
          count;

        bestToken =
          Number(token);
      }
    }


    if (
      !Number.isInteger(
        bestToken
      )
    ) {
      return null;
    }


    return {
      tokenId:
        bestToken,

      order:
        found.order,

      confidence:
        total > 0
          ? bestCount / total
          : 0,

      total
    };
  }


  /* =======================================================
     EMBEDDINGS
  ======================================================= */

  embedSequence(
    tokens
  ) {

    const d =
      this.config.embeddingSize;


    const context =
      tokens.length >
      this.config.contextSize
        ? tokens.slice(
            -this.config.contextSize
          )
        : tokens;


    const result =
      new Array(
        context.length
      );


    for (
      let position = 0;
      position < context.length;
      position++
    ) {

      let tokenId =
        Number(
          context[position]
        );


      if (
        !Number.isInteger(
          tokenId
        ) ||
        tokenId < 0 ||
        tokenId >=
          this.config.vocabSize
      ) {

        tokenId = 0;
      }


      const vector =
        new Float32Array(d);


      const tokenStart =
        tokenId * d;


      const posStart =
        position * d;


      for (
        let i = 0;
        i < d;
        i++
      ) {

        vector[i] =
          this.tokenEmbedding.data[
            tokenStart + i
          ] +
          this.positionEmbedding.data[
            posStart + i
          ];
      }


      result[position] =
        vector;
    }


    return result;
  }


  /* =======================================================
     FORWARD
  ======================================================= */

  forward(
    tokens
  ) {

    let hidden =
      this.embedSequence(
        tokens
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


    return hidden;
  }


  /* =======================================================
     OUTPUT LOGITS
  ======================================================= */

  logitsFromHidden(
    hidden
  ) {

    const vocab =
      this.config.vocabSize;

    const d =
      this.config.embeddingSize;


    const logits =
      new Float32Array(
        vocab
      );


    for (
      let tokenId = 0;
      tokenId < vocab;
      tokenId++
    ) {

      const start =
        tokenId * d;


      let value =
        this.outputBias.data[
          tokenId
        ];


      for (
        let i = 0;
        i < d;
        i++
      ) {

        value +=
          hidden[i] *
          this.tokenEmbedding.data[
            start + i
          ];
      }


      logits[tokenId] =
        value;
    }


    return logits;
  }


  /* =======================================================
     REPETITION PENALTY
  ======================================================= */

  applyRepetitionPenalty(
    logits,
    context
  ) {

    const penalty =
      Number(
        this.config.repetitionPenalty
      );


    if (
      penalty <= 1
    ) {
      return;
    }


    const recent =
      context.slice(
        -64
      );


    const seen =
      new Set();


    for (
      const tokenId
      of recent
    ) {

      const id =
        Number(tokenId);


      if (
        Number.isInteger(id) &&
        id >= 0 &&
        id < logits.length
      ) {

        seen.add(
          id
        );
      }
    }


    for (
      const tokenId
      of seen
    ) {

      if (
        logits[tokenId] > 0
      ) {

        logits[tokenId] /=
          penalty;

      } else {

        logits[tokenId] *=
          penalty;
      }
    }
  }


  /* =======================================================
     SAMPLING
  ======================================================= */

  sampleFromLogits(
    logits,
    options = {}
  ) {

    const temperature =
      Number.isFinite(
        options.temperature
      )
        ? options.temperature
        : this.config.temperature;


    const topK =
      Number.isInteger(
        options.topK
      )
        ? options.topK
        : this.config.topK;


    const topP =
      Number.isFinite(
        options.topP
      )
        ? options.topP
        : this.config.topP;


    const context =
      Array.isArray(
        options.previousTokens
      )
        ? options.previousTokens
        : [];


    this.applyRepetitionPenalty(
      logits,
      context
    );


    const probabilities =
      softmax(
        logits,
        temperature
      );


    let candidates =
      [];


    for (
      let i = 0;
      i < probabilities.length;
      i++
    ) {

      candidates.push({
        id: i,
        probability:
          probabilities[i]
      });
    }


    candidates.sort(
      (a, b) =>
        b.probability -
        a.probability
    );


    if (
      topK > 0 &&
      candidates.length > topK
    ) {

      candidates =
        candidates.slice(
          0,
          topK
        );
    }


    if (
      topP > 0 &&
      topP < 1
    ) {

      let sum =
        0;

      const filtered =
        [];


      for (
        const candidate
        of candidates
      ) {

        sum +=
          candidate.probability;

        filtered.push(
          candidate
        );


        if (
          sum >= topP
        ) {
          break;
        }
      }


      candidates =
        filtered;
    }


    let total =
      0;


    for (
      const candidate
      of candidates
    ) {

      total +=
        candidate.probability;
    }


    if (
      total <= 0 ||
      !Number.isFinite(total)
    ) {

      return (
        candidates[0]?.id ??
        0
      );
    }


    let random =
      this.rng() *
      total;


    for (
      const candidate
      of candidates
    ) {

      random -=
        candidate.probability;


      if (
        random <= 0
      ) {

        return candidate.id;
      }
    }


    return (
      candidates[
        candidates.length - 1
      ]?.id ??
      0
    );
  }


  /* =======================================================
     PREDICT NEXT TOKEN
  ======================================================= */

  predictNext(
    tokens,
    options = {}
  ) {

    const context =
      Array.from(
        tokens || []
      )
        .map(Number)
        .filter(
          id =>
            Number.isInteger(id) &&
            id >= 0 &&
            id <
              this.config.vocabSize
        );


    const sliced =
      context.length >
      this.config.contextSize
        ? context.slice(
            -this.config.contextSize
          )
        : context;


    /*
      1. Zuerst nach exakt gelernten
         N-Gram-Mustern suchen.

      Das ist besonders wichtig für
      deine kleinen JSON-Datensätze.
    */

    const ngram =
      this.ngramPrediction(
        sliced
      );


    const useMemory =
      options.useMemory !== false;


    if (
      useMemory &&
      ngram &&
      (
        ngram.order >= 4 ||
        ngram.confidence >= 0.75
      )
    ) {

      const logits =
        new Float32Array(
          this.config.vocabSize
        );


      logits[
        ngram.tokenId
      ] = 10;


      const probabilities =
        softmax(
          logits,
          1
        );


      return {
        token:
          ngram.tokenId,

        tokenId:
          ngram.tokenId,

        logits:
          Array.from(
            logits
          ),

        probabilities:
          Array.from(
            probabilities
          ),

        source:
          "memory",

        ngramOrder:
          ngram.order,

        confidence:
          ngram.confidence
      };
    }


    /*
      2. Neuronaler Fallback
    */

    if (
      sliced.length === 0
    ) {

      const logits =
        new Float32Array(
          this.config.vocabSize
        );


      const probabilities =
        softmax(
          logits,
          options.temperature ??
            this.config.temperature
        );


      const tokenId =
        this.sampleFromLogits(
          logits,
          {
            ...options,
            previousTokens:
              sliced
          }
        );


      return {
        token:
          tokenId,

        tokenId:
          tokenId,

        logits:
          Array.from(logits),

        probabilities:
          Array.from(probabilities),

        source:
          "neural"
      };
    }


    const hiddenStates =
      this.forward(
        sliced
      );


    const hidden =
      hiddenStates[
        hiddenStates.length - 1
      ];


    const logits =
      this.logitsFromHidden(
        hidden
      );


    const tokenId =
      this.sampleFromLogits(
        logits,
        {
          ...options,
          previousTokens:
            sliced
        }
      );


    const probabilities =
      softmax(
        logits,
        options.temperature ??
          this.config.temperature
      );


    return {
      token:
        tokenId,

      tokenId:
        tokenId,

      logits:
        Array.from(
          logits
        ),

      probabilities:
        Array.from(
          probabilities
        ),

      source:
        "neural"
    };
  }


  /* =======================================================
     TRAINING
  ======================================================= */

  trainStep(
    sequence
  ) {

    if (
      !Array.isArray(sequence) ||
      sequence.length < 2
    ) {

      return {
        loss: 0,
        tokens: 0
      };
    }


    const clean =
      sequence
        .map(Number)
        .filter(
          id =>
            Number.isInteger(id) &&
            id >= 0 &&
            id <
              this.config.vocabSize
        );


    if (
      clean.length < 2
    ) {

      return {
        loss: 0,
        tokens: 0
      };
    }


    /*
      Hauptlernen:
      N-Gram-Memory.

      Dadurch wird jede Trainingssequenz
      tatsächlich im Modell abgelegt.
    */

    this.rememberSequence(
      clean
    );


    /*
      Einfaches neuronales Online-Training.
    */

    const maxPositions =
      Math.min(
        clean.length - 1,
        32
      );


    let totalLoss =
      0;


    let trainedTokens =
      0;


    const stepSize =
      Math.max(
        1,
        Math.floor(
          (
            clean.length - 1
          ) /
          maxPositions
        )
      );


    for (
      let position = 0;
      position < clean.length - 1;
      position += stepSize
    ) {

      const contextStart =
        Math.max(
          0,
          position -
          this.config.contextSize +
          1
        );


      const context =
        clean.slice(
          contextStart,
          position + 1
        );


      const target =
        clean[
          position + 1
        ];


      const hiddenStates =
        this.forward(
          context
        );


      const hidden =
        hiddenStates[
          hiddenStates.length - 1
        ];


      const logits =
        this.logitsFromHidden(
          hidden
        );


      const probabilities =
        softmax(
          logits,
          1
        );


      const p =
        Math.max(
          1e-9,
          probabilities[target]
        );


      totalLoss +=
        -Math.log(p);


      /*
        Leichtes Output-Embedding-Update.

        Absichtlich begrenzt, damit bei großen
        JSON-Datensätzen keine NaNs entstehen.
      */

      const lr =
        this.getLearningRate();


      const d =
        this.config.embeddingSize;


      const targetStart =
        target * d;


      for (
        let i = 0;
        i < d;
        i++
      ) {

        const error =
          clamp(
            hidden[i] -
              this.tokenEmbedding.data[
                targetStart + i
              ],
            -1,
            1
          );


        this.tokenEmbedding.data[
          targetStart + i
        ] +=
          lr *
          0.02 *
          error;
      }


      this.outputBias.data[
        target
      ] +=
        lr *
        0.05;


      trainedTokens++;
    }


    this.trainingStep++;

    this.totalTokensSeen +=
      trainedTokens;


    this.lastLoss =
      trainedTokens > 0
        ? totalLoss /
          trainedTokens
        : 0;


    return {
      loss:
        this.lastLoss,

      tokens:
        trainedTokens,

      ngramEntries:
        this.ngrams.size
    };
  }


  trainBackprop(
    sequence
  ) {

    return this.trainStep(
      sequence
    );
  }


  /* =======================================================
     LEARNING RATE
  ======================================================= */

  getLearningRate() {

    const start =
      Number(
        this.config.learningRate
      );


    const minimum =
      Number(
        this.config.minLearningRate
      );


    const decay =
      Math.exp(
        -this.trainingStep /
        100000
      );


    return Math.max(
      minimum,
      start * decay
    );
  }


  /* =======================================================
     GENERIERUNG
  ======================================================= */

  generateTokens(
    promptTokens,
    options = {}
  ) {

    let tokens =
      Array.from(
        promptTokens || []
      )
        .map(Number)
        .filter(
          id =>
            Number.isInteger(id) &&
            id >= 0 &&
            id <
              this.config.vocabSize
        );


    const generated =
      [];


    const maxTokens =
      Math.min(
        Number.isInteger(
          options.maxTokens
        )
          ? options.maxTokens
          : 128,
        512
      );


    const stopTokens =
      new Set(
        Array.isArray(
          options.stopTokens
        )
          ? options.stopTokens
              .map(Number)
              .filter(
                Number.isInteger
              )
          : []
      );


    for (
      let step = 0;
      step < maxTokens;
      step++
    ) {

      const context =
        tokens.length >
        this.config.contextSize
          ? tokens.slice(
              -this.config.contextSize
            )
          : tokens;


      const result =
        this.predictNext(
          context,
          options
        );


      const tokenId =
        Number(
          result.tokenId
        );


      if (
        !Number.isInteger(
          tokenId
        ) ||
        tokenId < 0 ||
        tokenId >=
          this.config.vocabSize
      ) {

        break;
      }


      if (
        stopTokens.has(
          tokenId
        )
      ) {

        break;
      }


      tokens.push(
        tokenId
      );


      generated.push(
        tokenId
      );
    }


    return generated;
  }


  generate(
    promptTokens,
    options = {}
  ) {

    return this.generateTokens(
      promptTokens,
      options
    );
  }


  chat(
    promptTokens,
    options = {}
  ) {

    return this.generateTokens(
      promptTokens,
      options
    );
  }


  /* =======================================================
     PARAMETER
  ======================================================= */

  getParameters() {

    const parameters = [
      this.tokenEmbedding,
      this.positionEmbedding,
      this.outputBias
    ];


    for (
      const block of
      this.blocks
    ) {

      parameters.push(
        block.q,
        block.k,
        block.v,
        block.o,
        block.ff1,
        block.ff2,
        block.ffBias1,
        block.ffBias2,
        block.norm1.gamma,
        block.norm1.beta,
        block.norm2.gamma,
        block.norm2.beta
      );
    }


    return parameters;
  }


  /* =======================================================
     OPTIMIZER
  ======================================================= */

  optimize() {

    const learningRate =
      this.getLearningRate();


    const beta1 =
      0.9;

    const beta2 =
      0.999;

    const epsilon =
      1e-8;

    for (
      const parameter
      of this.getParameters()
    ) {

      const data =
        parameter.data;

      const grad =
        parameter.grad;


      for (
        let i = 0;
        i < data.length;
        i++
      ) {

        const g =
          Number.isFinite(
            grad[i]
          )
            ? clamp(
                grad[i],
                -1,
                1
              )
            : 0;


        parameter.m[i] =
          beta1 *
          parameter.m[i] +
          (1 - beta1) *
          g;


        parameter.v[i] =
          beta2 *
          parameter.v[i] +
          (1 - beta2) *
          g *
          g;


        const mHat =
          parameter.m[i] /
          (
            1 -
            Math.pow(
              beta1,
              this.trainingStep + 1
            )
          );


        const vHat =
          parameter.v[i] /
          (
            1 -
            Math.pow(
              beta2,
              this.trainingStep + 1
            )
          );


        data[i] -=
          learningRate *
          (
            mHat /
            (
              Math.sqrt(
                vHat
              ) +
              epsilon
            )
          );
      }


      parameter.zeroGrad();
    }
  }


  /* =======================================================
     SERIALISIERUNG
  ======================================================= */

  parameterToJSON(
    parameter
  ) {

    return {
      data:
        Array.from(
          parameter.data
        )
    };
  }


  parameterFromJSON(
    parameter,
    saved
  ) {

    if (
      !saved ||
      !Array.isArray(
        saved.data
      )
    ) {
      return;
    }


    const length =
      Math.min(
        parameter.data.length,
        saved.data.length
      );


    for (
      let i = 0;
      i < length;
      i++
    ) {

      const value =
        Number(
          saved.data[i]
        );


      parameter.data[i] =
        Number.isFinite(
          value
        )
          ? value
          : 0;
    }
  }


  serializeNgrams() {

    return Array.from(
      this.ngrams.entries()
    ).map(
      ([key, table]) => [
        key,
        Array.from(
          table.entries()
        )
      ]
    );
  }


  serialize() {

    return {
      version: 5,

      modelType:
        this.config.modelType,

      config:
        {
          ...this.config
        },

      trainingStep:
        this.trainingStep,

      totalTokensSeen:
        this.totalTokensSeen,

      lastLoss:
        this.lastLoss,

      loadedTrainingFiles:
        this.loadedTrainingFiles,

      loadedTrainingExamples:
        this.loadedTrainingExamples,

      tokenEmbedding:
        this.parameterToJSON(
          this.tokenEmbedding
        ),

      positionEmbedding:
        this.parameterToJSON(
          this.positionEmbedding
        ),

      outputBias:
        this.parameterToJSON(
          this.outputBias
        ),

      blocks:
        this.blocks.map(
          block => ({
            q:
              this.parameterToJSON(
                block.q
              ),

            k:
              this.parameterToJSON(
                block.k
              ),

            v:
              this.parameterToJSON(
                block.v
              ),

            o:
              this.parameterToJSON(
                block.o
              ),

            ff1:
              this.parameterToJSON(
                block.ff1
              ),

            ff2:
              this.parameterToJSON(
                block.ff2
              ),

            ffBias1:
              this.parameterToJSON(
                block.ffBias1
              ),

            ffBias2:
              this.parameterToJSON(
                block.ffBias2
              ),

            norm1Gamma:
              this.parameterToJSON(
                block.norm1.gamma
              ),

            norm1Beta:
              this.parameterToJSON(
                block.norm1.beta
              ),

            norm2Gamma:
              this.parameterToJSON(
                block.norm2.gamma
              ),

            norm2Beta:
              this.parameterToJSON(
                block.norm2.beta
              )
          })
        ),

      ngrams:
        this.serializeNgrams()
    };
  }


  toJSON() {
    return this.serialize();
  }


  /* =======================================================
     LOAD
  ======================================================= */

  load(data) {

    if (!data) {
      return this;
    }


    if (
      data.config &&
      typeof data.config ===
      "object"
    ) {

      const savedVocab =
        Number(
          data.config.vocabSize
        );


      if (
        savedVocab ===
        this.config.vocabSize
      ) {

        this.config = {
          ...this.config,
          ...data.config
        };
      }
    }


    this.parameterFromJSON(
      this.tokenEmbedding,
      data.tokenEmbedding
    );


    this.parameterFromJSON(
      this.positionEmbedding,
      data.positionEmbedding
    );


    this.parameterFromJSON(
      this.outputBias,
      data.outputBias
    );


    if (
      Array.isArray(
        data.blocks
      )
    ) {

      const count =
        Math.min(
          this.blocks.length,
          data.blocks.length
        );


      for (
        let i = 0;
        i < count;
        i++
      ) {

        const source =
          data.blocks[i];

        const target =
          this.blocks[i];


        this.parameterFromJSON(
          target.q,
          source.q
        );

        this.parameterFromJSON(
          target.k,
          source.k
        );

        this.parameterFromJSON(
          target.v,
          source.v
        );

        this.parameterFromJSON(
          target.o,
          source.o
        );

        this.parameterFromJSON(
          target.ff1,
          source.ff1
        );

        this.parameterFromJSON(
          target.ff2,
          source.ff2
        );

        this.parameterFromJSON(
          target.ffBias1,
          source.ffBias1
        );

        this.parameterFromJSON(
          target.ffBias2,
          source.ffBias2
        );

        this.parameterFromJSON(
          target.norm1.gamma,
          source.norm1Gamma
        );

        this.parameterFromJSON(
          target.norm1.beta,
          source.norm1Beta
        );

        this.parameterFromJSON(
          target.norm2.gamma,
          source.norm2Gamma
        );

        this.parameterFromJSON(
          target.norm2.beta,
          source.norm2Beta
        );
      }
    }


    this.trainingStep =
      Number.isInteger(
        data.trainingStep
      )
        ? data.trainingStep
        : 0;


    this.totalTokensSeen =
      Number.isInteger(
        data.totalTokensSeen
      )
        ? data.totalTokensSeen
        : 0;


    this.lastLoss =
      Number.isFinite(
        data.lastLoss
      )
        ? data.lastLoss
        : null;


    this.loadedTrainingFiles =
      Number.isInteger(
        data.loadedTrainingFiles
      )
        ? data.loadedTrainingFiles
        : 0;


    this.loadedTrainingExamples =
      Number.isInteger(
        data.loadedTrainingExamples
      )
        ? data.loadedTrainingExamples
        : 0;


    /* -------------------------
       N-GRAMS RESTORE
    ------------------------- */

    this.ngrams =
      new Map();


    if (
      Array.isArray(
        data.ngrams
      )
    ) {

      for (
        const entry of
        data.ngrams
      ) {

        if (
          !Array.isArray(
            entry
          ) ||
          entry.length !== 2
        ) {
          continue;
        }


        const key =
          String(
            entry[0]
          );


        const table =
          new Map();


        if (
          Array.isArray(
            entry[1]
          )
        ) {

          for (
            const pair
            of entry[1]
          ) {

            if (
              Array.isArray(pair) &&
              pair.length === 2
            ) {

              const token =
                Number(
                  pair[0]
                );

              const count =
                Number(
                  pair[1]
                );


              if (
                Number.isInteger(
                  token
                ) &&
                token >= 0 &&
                token <
                  this.config.vocabSize &&
                Number.isFinite(
                  count
                )
              ) {

                table.set(
                  token,
                  count
                );
              }
            }
          }
        }


        if (
          table.size > 0
        ) {

          this.ngrams.set(
            key,
            table
          );
        }
      }
    }


    return this;
  }


  /* =======================================================
     INFO
  ======================================================= */

  parameterCount() {

    let total = 0;


    for (
      const parameter
      of this.getParameters()
    ) {

      total +=
        parameter.data.length;
    }


    return total;
  }


  info() {

    return {
      modelType:
        this.config.modelType,

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

      parameters:
        this.parameterCount(),

      ngramEntries:
        this.ngrams.size,

      trainingExamples:
        this.loadedTrainingExamples,

      trainingFiles:
        this.loadedTrainingFiles,

      trainingStep:
        this.trainingStep,

      totalTokensSeen:
        this.totalTokensSeen,

      lastLoss:
        this.lastLoss
    };
  }
}


/* =========================================================
   MODEL VARIANTEN
========================================================= */

class SmallLanguageModel
  extends LanguageModel {

  constructor(config = {}) {

    super({
      ...DEFAULT_CONFIG,

      contextSize: 128,
      embeddingSize: 64,
      layers: 2,
      heads: 2,
      headSize: 32,
      feedForwardSize: 256,

      ...config,

      modelType:
        "LUMORA-SMALL"
    });
  }
}


class LargeLanguageModel
  extends LanguageModel {

  constructor(config = {}) {

    super({
      ...DEFAULT_CONFIG,

      contextSize: 256,
      embeddingSize: 128,
      layers: 4,
      heads: 4,
      headSize: 32,
      feedForwardSize: 512,

      ...config,

      modelType:
        "LUMORA-LARGE"
    });
  }
}


/* =========================================================
   CHAT ENGINE
========================================================= */

class ChatEngine {

  constructor(model) {

    this.model =
      model ||
      new LargeLanguageModel();
  }


  predictNext(
    tokens,
    options = {}
  ) {

    return this.model.predictNext(
      tokens,
      options
    );
  }


  generate(
    tokens,
    options = {}
  ) {

    return this.model.generate(
      tokens,
      options
    );
  }


  chat(
    tokens,
    options = {}
  ) {

    return this.model.chat(
      tokens,
      options
    );
  }
}


/* =========================================================
   CONFIG EXPORT
========================================================= */

const MODEL_CONFIG = {
  ...DEFAULT_CONFIG,

  small: {
    contextSize: 128,
    embeddingSize: 64,
    layers: 2,
    heads: 2,
    headSize: 32,
    feedForwardSize: 256
  },

  large: {
    contextSize: 256,
    embeddingSize: 128,
    layers: 4,
    heads: 4,
    headSize: 32,
    feedForwardSize: 512
  },

  bigger: {
    contextSize: 256,
    embeddingSize: 256,
    layers: 6,
    heads: 8,
    headSize: 32,
    feedForwardSize: 1024
  }
};


/* =========================================================
   EXPORTS
========================================================= */

module.exports = {
  LanguageModel,
  SmallLanguageModel,
  LargeLanguageModel,
  ChatEngine,
  Parameter,
  MODEL_CONFIG
};

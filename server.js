"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const url = require("url");
const { Worker } = require("worker_threads");

const ROOT = __dirname;

const DIR = {
    daten: path.join(ROOT, "DATEN"),
    gelernt: path.join(ROOT, "GELERNT")
};

const FILE = {
    model: path.join(DIR.gelernt, "model.json"),
    tokenizer: path.join(DIR.gelernt, "tokenizer.json"),
    config: path.join(DIR.gelernt, "config.json"),
    state: path.join(DIR.gelernt, "training-state.json")
};

const PORT = Number(process.env.PORT || 3000);

const MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon"
};

const STATUS = {
    server: "starting",
    data: {
        loading: false,
        loaded: false,
        files: [],
        examples: 0,
        bytes: 0
    },
    model: {
        loading: false,
        loaded: false,
        exists: false,
        size: 0
    },
    tokenizer: {
        loading: false,
        loaded: false,
        exists: false,
        size: 0
    },
    training: {
        running: false,
        epoch: 0,
        step: 0,
        loss: null,
        startedAt: null,
        finishedAt: null,
        error: null
    }
};

let worker = null;
let modelCache = null;
let tokenizerCache = null;


/* =========================================================
   ORDNER AUTOMATISCH ERSTELLEN
   ========================================================= */

function createDirectories() {
    fs.mkdirSync(DIR.daten, {
        recursive: true
    });

    fs.mkdirSync(DIR.gelernt, {
        recursive: true
    });
}


/* =========================================================
   LEERE DATEIEN
   ========================================================= */

function createInitialFiles() {
    if (!fs.existsSync(FILE.config)) {
        atomicWrite(
            FILE.config,
            JSON.stringify({
                version: 1,
                createdAt: new Date().toISOString(),
                model: {},
                training: {}
            }, null, 2)
        );
    }

    if (!fs.existsSync(FILE.state)) {
        atomicWrite(
            FILE.state,
            JSON.stringify({
                version: 1,
                epoch: 0,
                step: 0,
                loss: null,
                sourceFiles: [],
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
            }, null, 2)
        );
    }
}


/* =========================================================
   ATOMISCH SPEICHERN
   ========================================================= */

function atomicWrite(file, data) {
    const temp =
        file + ".tmp-" + process.pid + "-" + Date.now();

    fs.writeFileSync(
        temp,
        data,
        "utf8"
    );

    fs.renameSync(
        temp,
        file
    );
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
   DATEN SCANNEN
   ========================================================= */

function scanDataFolder() {
    if (!fs.existsSync(DIR.daten)) {
        return [];
    }

    return fs.readdirSync(
        DIR.daten,
        {
            withFileTypes: true
        }
    )
    .filter(entry =>
        entry.isFile() &&
        entry.name.toLowerCase().endsWith(".json")
    )
    .map(entry => {
        const file =
            path.join(
                DIR.daten,
                entry.name
            );

        const stat =
            fs.statSync(file);

        return {
            name: entry.name,
            size: stat.size,
            modified:
                stat.mtime.toISOString()
        };
    });
}


/* =========================================================
   TEXT AUS JSON HOLEN
   ========================================================= */

function collectText(
    value,
    output
) {
    if (
        typeof value ===
        "string"
    ) {
        const text =
            value.trim();

        if (text) {
            output.push(text);
        }

        return;
    }

    if (
        Array.isArray(value)
    ) {
        for (
            const item of value
        ) {
            collectText(
                item,
                output
            );
        }

        return;
    }

    if (
        value &&
        typeof value ===
        "object"
    ) {
        for (
            const key of
            Object.keys(value)
        ) {
            collectText(
                value[key],
                output
            );
        }
    }
}


/* =========================================================
   FRAGE / ANTWORT ERKENNEN
   ========================================================= */

const QUESTION_KEYS = [
    "frage",
    "question",
    "user",
    "input",
    "prompt"
];

const ANSWER_KEYS = [
    "antwort",
    "answer",
    "assistant",
    "output",
    "response"
];

function findKey(
    object,
    keys
) {
    if (
        !object ||
        typeof object !==
        "object"
    ) {
        return null;
    }

    const names =
        Object.keys(object);

    for (
        const key of names
    ) {
        if (
            keys.includes(
                key.toLowerCase()
            )
        ) {
            return key;
        }
    }

    return null;
}


function collectExamples(
    value,
    output
) {
    if (
        Array.isArray(value)
    ) {
        for (
            const item of value
        ) {
            collectExamples(
                item,
                output
            );
        }

        return;
    }

    if (
        !value ||
        typeof value !==
        "object"
    ) {
        return;
    }

    const qKey =
        findKey(
            value,
            QUESTION_KEYS
        );

    const aKey =
        findKey(
            value,
            ANSWER_KEYS
        );

    if (
        qKey &&
        aKey &&
        typeof value[qKey] ===
        "string" &&
        typeof value[aKey] ===
        "string"
    ) {
        output.push({
            question:
                value[qKey].trim(),

            answer:
                value[aKey].trim()
        });
    }

    for (
        const key of
        Object.keys(value)
    ) {
        collectExamples(
            value[key],
            output
        );
    }
}


/* =========================================================
   DATEN LADEN
   ========================================================= */

function loadTrainingData() {
    STATUS.data.loading = true;

    const files =
        scanDataFolder();

    const texts = [];
    const examples = [];
    let bytes = 0;

    for (
        const info of files
    ) {
        const file =
            path.join(
                DIR.daten,
                info.name
            );

        try {
            const raw =
                fs.readFileSync(
                    file,
                    "utf8"
                );

            bytes +=
                Buffer.byteLength(
                    raw,
                    "utf8"
                );

            try {
                const json =
                    JSON.parse(raw);

                collectExamples(
                    json,
                    examples
                );

                collectText(
                    json,
                    texts
                );

            } catch {
                if (raw.trim()) {
                    texts.push(
                        raw.trim()
                    );
                }
            }

        } catch (error) {
            console.error(
                "DATEN Fehler:",
                info.name,
                error.message
            );
        }
    }

    const uniqueExamples =
        [];

    const seen =
        new Set();

    for (
        const example of
        examples
    ) {
        const key =
            example.question +
            "\n" +
            example.answer;

        if (
            !seen.has(key)
        ) {
            seen.add(key);

            uniqueExamples.push(
                example
            );
        }
    }

    STATUS.data.loading = false;
    STATUS.data.loaded = true;
    STATUS.data.files = files;
    STATUS.data.examples =
        uniqueExamples.length;
    STATUS.data.bytes = bytes;

    return {
        files,
        examples:
            uniqueExamples,
        texts,
        bytes
    };
}


/* =========================================================
   GELERNT SCANNEN
   ========================================================= */

function scanLearned() {
    const result = {};

    for (
        const key of
        ["model", "tokenizer", "config", "state"]
    ) {
        const file =
            FILE[key];

        const exists =
            fs.existsSync(file);

        result[key] = {
            exists,
            size: exists
                ? fs.statSync(file).size
                : 0
        };
    }

    STATUS.model.exists =
        result.model.exists;

    STATUS.model.size =
        result.model.size;

    STATUS.tokenizer.exists =
        result.tokenizer.exists;

    STATUS.tokenizer.size =
        result.tokenizer.size;

    return result;
}


/* =========================================================
   GELERNT IM HINTERGRUND LADEN
   ========================================================= */

async function backgroundLoadLearned() {
    STATUS.model.loading = true;
    STATUS.tokenizer.loading = true;

    try {
        if (fs.existsSync(FILE.model)) {
            try {
                modelCache =
                    readJSON(
                        FILE.model
                    );

                STATUS.model.loaded =
                    true;

            } catch (error) {
                console.error(
                    "Modelldatei fehlerhaft:",
                    error.message
                );
            }
        }

        if (fs.existsSync(FILE.tokenizer)) {
            try {
                tokenizerCache =
                    readJSON(
                        FILE.tokenizer
                    );

                STATUS.tokenizer.loaded =
                    true;

            } catch (error) {
                console.error(
                    "Tokenizerdatei fehlerhaft:",
                    error.message
                );
            }
        }

    } finally {
        STATUS.model.loading = false;
        STATUS.tokenizer.loading = false;
    }
}


/* =========================================================
   BODY LESEN
   ========================================================= */

function readBody(req) {
    return new Promise(
        (resolve, reject) => {
            const chunks = [];

            let length = 0;

            req.on(
                "data",
                chunk => {
                    length +=
                        chunk.length;

                    if (
                        length >
                        20 * 1024 * 1024
                    ) {
                        reject(
                            new Error(
                                "Request zu groß."
                            )
                        );

                        req.destroy();
                        return;
                    }

                    chunks.push(chunk);
                }
            );

            req.on(
                "end",
                () => {
                    resolve(
                        Buffer.concat(
                            chunks
                        ).toString(
                            "utf8"
                        )
                    );
                }
            );

            req.on(
                "error",
                reject
            );
        }
    );
}


/* =========================================================
   JSON RESPONSE
   ========================================================= */

function sendJSON(
    res,
    statusCode,
    data
) {
    const body =
        JSON.stringify(
            data
        );

    res.writeHead(
        statusCode,
        {
            "Content-Type":
                "application/json; charset=utf-8",

            "Cache-Control":
                "no-store",

            "Access-Control-Allow-Origin":
                "*"
        }
    );

    res.end(
        body
    );
}


/* =========================================================
   TEXT RESPONSE
   ========================================================= */

function sendText(
    res,
    statusCode,
    text
) {
    res.writeHead(
        statusCode,
        {
            "Content-Type":
                "text/plain; charset=utf-8",

            "Access-Control-Allow-Origin":
                "*"
        }
    );

    res.end(
        text
    );
}


/* =========================================================
   STATIC FILES
   ========================================================= */

function serveStatic(
    req,
    res,
    pathname
) {
    let requested =
        decodeURIComponent(
            pathname
        );

    if (
        requested ===
        "/"
    ) {
        requested =
            "/index.html";
    }

    const fullPath =
        path.resolve(
            ROOT,
            "." +
            requested
        );

    if (
        !fullPath.startsWith(
            ROOT
        )
    ) {
        sendText(
            res,
            403,
            "Forbidden"
        );

        return;
    }

    if (
        !fs.existsSync(
            fullPath
        )
    ) {
        sendText(
            res,
            404,
            "Not found"
        );

        return;
    }

    const stat =
        fs.statSync(
            fullPath
        );

    if (
        !stat.isFile()
    ) {
        sendText(
            res,
            404,
            "Not found"
        );

        return;
    }

    const extension =
        path.extname(
            fullPath
        ).toLowerCase();

    res.writeHead(
        200,
        {
            "Content-Type":
                MIME[extension] ||
                "application/octet-stream",

            "Cache-Control":
                "no-cache"
        }
    );

    fs.createReadStream(
        fullPath
    ).pipe(
        res
    );
}


/* =========================================================
   API
   ========================================================= */

async function handleAPI(
    req,
    res,
    pathname
) {

    if (
        pathname ===
        "/api/status"
    ) {
        scanLearned();

        sendJSON(
            res,
            200,
            {
                ...STATUS,

                learned:
                    scanLearned(),

                time:
                    new Date().toISOString()
            }
        );

        return true;
    }


    if (
        pathname ===
        "/api/files"
    ) {
        const data =
            loadTrainingData();

        sendJSON(
            res,
            200,
            {
                files:
                    data.files,

                examples:
                    data.examples.length,

                bytes:
                    data.bytes
            }
        );

        return true;
    }


    if (
        pathname ===
        "/api/data"
    ) {
        const data =
            loadTrainingData();

        sendJSON(
            res,
            200,
            data
        );

        return true;
    }


    if (
        pathname ===
        "/api/model"
    ) {
        if (
            !fs.existsSync(
                FILE.model
            )
        ) {
            sendJSON(
                res,
                404,
                {
                    error:
                        "model.json nicht vorhanden."
                }
            );

            return true;
        }

        res.writeHead(
            200,
            {
                "Content-Type":
                    "application/json; charset=utf-8",

                "Cache-Control":
                    "no-store"
            }
        );

        fs.createReadStream(
            FILE.model
        ).pipe(
            res
        );

        return true;
    }


    if (
        pathname ===
        "/api/tokenizer"
    ) {
        if (
            !fs.existsSync(
                FILE.tokenizer
            )
        ) {
            sendJSON(
                res,
                404,
                {
                    error:
                        "tokenizer.json nicht vorhanden."
                }
            );

            return true;
        }

        res.writeHead(
            200,
            {
                "Content-Type":
                    "application/json; charset=utf-8",

                "Cache-Control":
                    "no-store"
            }
        );

        fs.createReadStream(
            FILE.tokenizer
        ).pipe(
            res
        );

        return true;
    }


    if (
        pathname ===
        "/api/config"
    ) {
        if (
            !fs.existsSync(
                FILE.config
            )
        ) {
            sendJSON(
                res,
                404,
                {
                    error:
                        "config.json nicht vorhanden."
                }
            );

            return true;
        }

        sendJSON(
            res,
            200,
            readJSON(
                FILE.config
            )
        );

        return true;
    }


    if (
        pathname ===
        "/api/training-state"
    ) {
        if (
            !fs.existsSync(
                FILE.state
            )
        ) {
            sendJSON(
                res,
                200,
                {}
            );

            return true;
        }

        sendJSON(
            res,
            200,
            readJSON(
                FILE.state
            )
        );

        return true;
    }


    if (
        pathname ===
        "/api/reload"
    ) {
        await backgroundLoadLearned();

        sendJSON(
            res,
            200,
            {
                ok: true,
                status: STATUS
            }
        );

        return true;
    }


    if (
        pathname ===
        "/api/train/start"
    ) {

        if (
            req.method !==
            "POST"
        ) {
            sendJSON(
                res,
                405,
                {
                    error:
                        "POST erforderlich."
                }
            );

            return true;
        }

        if (
            STATUS.training.running
        ) {
            sendJSON(
                res,
                409,
                {
                    error:
                        "Training läuft bereits."
                }
            );

            return true;
        }

        let options = {};

        try {
            const body =
                await readBody(
                    req
                );

            if (body.trim()) {
                options =
                    JSON.parse(
                        body
                    );
            }

        } catch (error) {
            sendJSON(
                res,
                400,
                {
                    error:
                        error.message
                }
            );

            return true;
        }

        startTraining(
            options
        );

        sendJSON(
            res,
            202,
            {
                ok: true,
                message:
                    "Training im Hintergrund gestartet."
            }
        );

        return true;
    }


    if (
        pathname ===
        "/api/train/stop"
    ) {

        if (
            worker
        ) {
            worker.postMessage({
                type:
                    "stop"
            });
        }

        sendJSON(
            res,
            200,
            {
                ok: true
            }
        );

        return true;
    }


    if (
        pathname ===
        "/api/generate"
    ) {

        if (
            req.method !==
            "POST"
        ) {
            sendJSON(
                res,
                405,
                {
                    error:
                        "POST erforderlich."
                }
            );

            return true;
        }

        let body;

        try {
            body =
                JSON.parse(
                    await readBody(
                        req
                    )
                );

        } catch {
            sendJSON(
                res,
                400,
                {
                    error:
                        "Ungültiges JSON."
                }
            );

            return true;
        }

        /*
         * Der eigentliche Generator kann
         * später hier in einem Worker laufen.
         */

        sendJSON(
            res,
            501,
            {
                error:
                    "Generator-Worker wird als nächstes angebunden.",
                prompt:
                    body.prompt || ""
            }
        );

        return true;
    }


    return false;
}


/* =========================================================
   TRAINING WORKER
   ========================================================= */

function startTraining(
    options
) {
    if (
        STATUS.training.running
    ) {
        return;
    }

    STATUS.training.running =
        true;

    STATUS.training.epoch =
        0;

    STATUS.training.step =
        0;

    STATUS.training.loss =
        null;

    STATUS.training.error =
        null;

    STATUS.training.startedAt =
        new Date().toISOString();

    STATUS.training.finishedAt =
        null;


    const workerFile =
        path.join(
            ROOT,
            "train-worker.js"
        );


    if (
        !fs.existsSync(
            workerFile
        )
    ) {
        STATUS.training.running =
            false;

        STATUS.training.error =
            "train-worker.js fehlt.";

        return;
    }


    worker =
        new Worker(
            workerFile,
            {
                workerData: {
                    root:
                        ROOT,

                    daten:
                        DIR.daten,

                    gelernt:
                        DIR.gelernt,

                    options
                }
            }
        );


    worker.on(
        "message",
        message => {

            if (
                message.type ===
                "progress"
            ) {
                STATUS.training.epoch =
                    message.epoch ??
                    STATUS.training.epoch;

                STATUS.training.step =
                    message.step ??
                    STATUS.training.step;

                STATUS.training.loss =
                    message.loss ??
                    STATUS.training.loss;
            }


            if (
                message.type ===
                "finished"
            ) {
                STATUS.training.running =
                    false;

                STATUS.training.finishedAt =
                    new Date().toISOString();

                modelCache =
                    null;

                tokenizerCache =
                    null;

                scanLearned();

                backgroundLoadLearned();
            }


            if (
                message.type ===
                "error"
            ) {
                STATUS.training.running =
                    false;

                STATUS.training.error =
                    message.error;
            }
        }
    );


    worker.on(
        "error",
        error => {

            STATUS.training.running =
                false;

            STATUS.training.error =
                error.message;
        }
    );


    worker.on(
        "exit",
        () => {

            worker =
                null;
        }
    );
}


/* =========================================================
   SERVER
   ========================================================= */

async function requestHandler(
    req,
    res
) {
    const parsed =
        url.parse(
            req.url,
            true
        );

    const pathname =
        parsed.pathname;


    if (
        req.method ===
        "OPTIONS"
    ) {
        res.writeHead(
            204,
            {
                "Access-Control-Allow-Origin":
                    "*",

                "Access-Control-Allow-Methods":
                    "GET,POST,OPTIONS",

                "Access-Control-Allow-Headers":
                    "Content-Type"
            }
        );

        res.end();

        return;
    }


    if (
        pathname.startsWith(
            "/api/"
        )
    ) {

        try {

            const handled =
                await handleAPI(
                    req,
                    res,
                    pathname
                );

            if (handled) {
                return;
            }

        } catch (error) {

            console.error(
                "API Fehler:",
                error
            );

            sendJSON(
                res,
                500,
                {
                    error:
                        error.message
                }
            );

            return;
        }
    }


    serveStatic(
        req,
        res,
        pathname
    );
}


/* =========================================================
   START
   ========================================================= */

createDirectories();

createInitialFiles();

STATUS.server =
    "starting";


const server =
    http.createServer(
        requestHandler
    );


server.listen(
    PORT,
    "0.0.0.0",
    () => {

        STATUS.server =
            "running";

        console.log("");
        console.log(
            "=========================================="
        );
        console.log(
            "          MEINE-KI SERVER"
        );
        console.log(
            "=========================================="
        );
        console.log(
            `http://localhost:${PORT}`
        );
        console.log("");
        console.log(
            "DATEN:   " +
            DIR.daten
        );
        console.log(
            "GELERNT: " +
            DIR.gelernt
        );
        console.log("");
        console.log(
            "Server läuft."
        );
        console.log(
            "Laden der KI läuft im Hintergrund."
        );
        console.log(
            "=========================================="
        );
        console.log("");

        /*
         * Erst NACH dem Serverstart laden.
         * Dadurch muss der Browser nicht warten,
         * bis große Modelldateien geladen wurden.
         */

        setImmediate(
            async () => {

                try {

                    loadTrainingData();

                    scanLearned();

                    await backgroundLoadLearned();

                    console.log(
                        "Hintergrund-Laden abgeschlossen."
                    );

                } catch (error) {

                    console.error(
                        "Hintergrundfehler:",
                        error
                    );
                }
            }
        );
    }
);


/* =========================================================
   SAUBER BEENDEN
   ========================================================= */

function shutdown() {

    console.log(
        "\nServer wird beendet..."
    );


    if (worker) {

        worker.postMessage({
            type:
                "stop"
        });
    }


    server.close(
        () => {

            process.exit(
                0
            );
        }
    );
}


process.on(
    "SIGINT",
    shutdown
);

process.on(
    "SIGTERM",
    shutdown
);


/* =========================================================
   EXPORT
   ========================================================= */

module.exports = {
    ROOT,
    DIR,
    FILE,
    STATUS,
    scanDataFolder,
    loadTrainingData,
    scanLearned,
    startTraining
};

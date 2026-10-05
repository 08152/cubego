"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const https = require("https");
const { Worker } = require("worker_threads");

const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, "DATEN");
const LEARNED_DIR = path.join(ROOT, "GELERNT");

const MODEL_FILE = path.join(LEARNED_DIR, "model.json");
const TOKENIZER_FILE = path.join(LEARNED_DIR, "tokenizer.json");
const CONFIG_FILE = path.join(LEARNED_DIR, "config.json");
const STATE_FILE = path.join(LEARNED_DIR, "training-state.json");

const PORT = Number(process.env.PORT || 10000);

const GITHUB_TOKEN = process.env.GITHUB_TOKEN || "";
const GITHUB_OWNER = process.env.GITHUB_OWNER || "08152";
const GITHUB_REPO = process.env.GITHUB_REPO || "cubego";
const GITHUB_BRANCH = process.env.GITHUB_BRANCH || "main";

const TRAIN_WORKER = path.join(
    ROOT,
    "train-worker.js"
);

const GENERATE_WORKER = path.join(
    ROOT,
    "generate-worker.js"
);

fs.mkdirSync(DATA_DIR, {
    recursive: true
});

fs.mkdirSync(LEARNED_DIR, {
    recursive: true
});

let trainingWorker = null;
let generateWorker = null;

let trainingState = {
    running: false,
    phase: "idle",
    progress: 0,
    epoch: 0,
    step: 0,
    loss: null,
    totalEpochs: 0,
    message: "Bereit."
};

let generationState = {
    running: false,
    text: "",
    error: null,
    startedAt: null,
    finishedAt: null
};


/* =========================================================
   HILFSFUNKTIONEN
   ========================================================= */

function json(res, status, data) {
    const body = JSON.stringify(
        data,
        null,
        2
    );

    res.writeHead(
        status,
        {
            "Content-Type":
                "application/json; charset=utf-8",

            "Cache-Control":
                "no-store",

            "Access-Control-Allow-Origin":
                "*",

            "Access-Control-Allow-Headers":
                "Content-Type",

            "Access-Control-Allow-Methods":
                "GET,POST,OPTIONS"
        }
    );

    res.end(body);
}

function text(res, status, value) {
    res.writeHead(
        status,
        {
            "Content-Type":
                "text/plain; charset=utf-8",

            "Cache-Control":
                "no-store",

            "Access-Control-Allow-Origin":
                "*"
        }
    );

    res.end(value);
}

function sendError(
    res,
    status,
    message,
    error
) {
    json(
        res,
        status,
        {
            success: false,
            error: message,
            details:
                error
                    ? error.message
                    : undefined
        }
    );
}

function readBody(req) {
    return new Promise(
        (resolve, reject) => {
            let body = "";

            req.on(
                "data",
                chunk => {
                    body += chunk;

                    if (
                        body.length >
                        10 * 1024 * 1024
                    ) {
                        reject(
                            new Error(
                                "Request zu groß."
                            )
                        );

                        req.destroy();
                    }
                }
            );

            req.on(
                "end",
                () => {
                    if (!body) {
                        resolve({});
                        return;
                    }

                    try {
                        resolve(
                            JSON.parse(
                                body
                            )
                        );
                    } catch {
                        reject(
                            new Error(
                                "Ungültiges JSON."
                            )
                        );
                    }
                }
            );

            req.on(
                "error",
                reject
            );
        }
    );
}

function fileExists(file) {
    try {
        return fs.existsSync(file);
    } catch {
        return false;
    }
}

function readJSON(file, fallback = null) {
    try {
        return JSON.parse(
            fs.readFileSync(
                file,
                "utf8"
            )
        );
    } catch {
        return fallback;
    }
}

function getDataFiles() {
    if (!fileExists(DATA_DIR)) {
        return [];
    }

    return fs.readdirSync(
        DATA_DIR,
        {
            withFileTypes: true
        }
    )
    .filter(entry => {
        if (!entry.isFile()) {
            return false;
        }

        const name =
            entry.name.toLowerCase();

        return (
            name.endsWith(".json") ||
            name.endsWith(".jsonl") ||
            name.endsWith(".txt")
        );
    })
    .map(entry =>
        entry.name
    );
}


/* =========================================================
   GITHUB
   ========================================================= */

function githubRequest(
    method,
    apiPath,
    body = null
) {
    return new Promise(
        (resolve, reject) => {
            if (!GITHUB_TOKEN) {
                reject(
                    new Error(
                        "GITHUB_TOKEN fehlt."
                    )
                );

                return;
            }

            const payload =
                body === null
                    ? null
                    : JSON.stringify(
                          body
                      );

            const options = {
                hostname:
                    "api.github.com",

                port:
                    443,

                path:
                    apiPath,

                method,

                headers: {
                    "User-Agent":
                        "LUMORA",

                    "Accept":
                        "application/vnd.github+json",

                    "Authorization":
                        "Bearer " +
                        GITHUB_TOKEN,

                    "X-GitHub-Api-Version":
                        "2022-11-28"
                }
            };

            if (payload) {
                options.headers[
                    "Content-Type"
                ] =
                    "application/json";

                options.headers[
                    "Content-Length"
                ] =
                    Buffer.byteLength(
                        payload
                    );
            }

            const request =
                https.request(
                    options,
                    response => {
                        let data = "";

                        response.on(
                            "data",
                            chunk => {
                                data += chunk;
                            }
                        );

                        response.on(
                            "end",
                            () => {
                                let parsed =
                                    data;

                                try {
                                    parsed =
                                        data
                                            ? JSON.parse(
                                                  data
                                              )
                                            : {};
                                } catch {}

                                if (
                                    response.statusCode >=
                                        200 &&
                                    response.statusCode <
                                        300
                                ) {
                                    resolve(
                                        parsed
                                    );
                                } else {
                                    reject(
                                        new Error(
                                            "GitHub API " +
                                            response.statusCode +
                                            ": " +
                                            (
                                                parsed.message ||
                                                data
                                            )
                                        )
                                    );
                                }
                            }
                        );
                    }
                );

            request.on(
                "error",
                reject
            );

            if (payload) {
                request.write(
                    payload
                );
            }

            request.end();
        }
    );
}

async function getGitHubFile(
    filePath
) {
    const api =
        "/repos/" +
        encodeURIComponent(
            GITHUB_OWNER
        ) +
        "/" +
        encodeURIComponent(
            GITHUB_REPO
        ) +
        "/contents/" +
        filePath
            .split("/")
            .map(
                encodeURIComponent
            )
            .join("/") +
        "?ref=" +
        encodeURIComponent(
            GITHUB_BRANCH
        );

    return githubRequest(
        "GET",
        api
    );
}

async function uploadGitHubFile(
    filePath,
    buffer,
    message
) {
    const encoded =
        Buffer.from(
            buffer
        ).toString(
            "base64"
        );

    let sha = null;

    try {
        const existing =
            await getGitHubFile(
                filePath
            );

        sha =
            existing.sha ||
            null;
    } catch {}

    const api =
        "/repos/" +
        encodeURIComponent(
            GITHUB_OWNER
        ) +
        "/" +
        encodeURIComponent(
            GITHUB_REPO
        ) +
        "/contents/" +
        filePath
            .split("/")
            .map(
                encodeURIComponent
            )
            .join("/");

    const body = {
        message:
            message ||
            "LUMORA: gelernte Dateien aktualisiert",

        content:
            encoded,

        branch:
            GITHUB_BRANCH
    };

    if (sha) {
        body.sha = sha;
    }

    return githubRequest(
        "PUT",
        api,
        body
    );
}

async function syncLearnedToGitHub() {
    if (!GITHUB_TOKEN) {
        return {
            success: false,
            message:
                "GITHUB_TOKEN fehlt."
        };
    }

    const files = [
        "model.json",
        "tokenizer.json",
        "config.json",
        "training-state.json"
    ];

    const uploaded = [];

    for (const filename of files) {
        const local =
            path.join(
                LEARNED_DIR,
                filename
            );

        if (!fileExists(local)) {
            continue;
        }

        const buffer =
            fs.readFileSync(
                local
            );

        await uploadGitHubFile(
            "GELERNT/" +
                filename,
            buffer,
            "LUMORA: " +
                filename +
                " aktualisiert"
        );

        uploaded.push(
            filename
        );
    }

    return {
        success: true,
        uploaded
    };
}


/* =========================================================
   STATUS
   ========================================================= */

function getModelStatus() {
    return {
        model:
            fileExists(
                MODEL_FILE
            ),

        tokenizer:
            fileExists(
                TOKENIZER_FILE
            ),

        config:
            fileExists(
                CONFIG_FILE
            ),

        modelFile:
            MODEL_FILE,

        tokenizerFile:
            TOKENIZER_FILE
    };
}

function getStatus() {
    return {
        success: true,

        name:
            "LUMORA",

        online:
            true,

        port:
            PORT,

        github: {
            active:
                Boolean(
                    GITHUB_TOKEN
                ),

            owner:
                GITHUB_OWNER,

            repo:
                GITHUB_REPO,

            branch:
                GITHUB_BRANCH
        },

        daten: {
            folder:
                DATA_DIR,

            files:
                getDataFiles(),

            count:
                getDataFiles().length
        },

        model:
            getModelStatus(),

        training:
            trainingState,

        generation:
            generationState
    };
}


/* =========================================================
   TRAINING
   ========================================================= */

function startTraining(
    payload
) {
    if (trainingWorker) {
        return {
            success: false,
            message:
                "Training läuft bereits."
        };
    }

    if (!fileExists(
        TRAIN_WORKER
    )) {
        return {
            success: false,
            message:
                "train-worker.js wurde nicht gefunden."
        };
    }

    const options = {
        epochs:
            Number(
                payload.epochs ||
                10
            ),

        sequenceLength:
            Number(
                payload.sequenceLength ||
                256
            ),

        learningRate:
            Number(
                payload.learningRate ||
                0.0003
            ),

        contextSize:
            Number(
                payload.contextSize ||
                256
            )
    };

    trainingState = {
        running: true,
        phase: "starting",
        progress: 0,
        epoch: 0,
        step: 0,
        loss: null,
        totalEpochs:
            options.epochs,
        message:
            "Training wird gestartet..."
    };

    trainingWorker =
        new Worker(
            TRAIN_WORKER,
            {
                workerData: {
                    root:
                        ROOT,

                    daten:
                        DATA_DIR,

                    gelernt:
                        LEARNED_DIR,

                    options
                }
            }
        );

    trainingWorker.on(
        "message",
        async message => {
            if (!message) {
                return;
            }

            if (
                message.type ===
                "started"
            ) {
                trainingState =
                    Object.assign(
                        {},
                        trainingState,
                        message,
                        {
                            running:
                                true
                        }
                    );

                return;
            }

            if (
                message.type ===
                "progress"
            ) {
                trainingState =
                    Object.assign(
                        {},
                        trainingState,
                        message,
                        {
                            running:
                                true
                        }
                    );

                return;
            }

            if (
                message.type ===
                "epoch"
            ) {
                trainingState =
                    Object.assign(
                        {},
                        trainingState,
                        message,
                        {
                            running:
                                true,

                            phase:
                                "training"
                        }
                    );

                return;
            }

            if (
                message.type ===
                "finished"
            ) {
                trainingState =
                    Object.assign(
                        {},
                        trainingState,
                        message,
                        {
                            running:
                                false,

                            phase:
                                "finished",

                            progress:
                                100,

                            message:
                                "Training abgeschlossen."
                        }
                    );

                const worker =
                    trainingWorker;

                trainingWorker =
                    null;

                try {
                    const github =
                        await syncLearnedToGitHub();

                    trainingState.github =
                        github;
                } catch (error) {
                    trainingState.github = {
                        success:
                            false,

                        message:
                            error.message
                    };
                }

                if (
                    worker
                ) {
                    try {
                        await worker.terminate();
                    } catch {}
                }

                return;
            }

            if (
                message.type ===
                "stopped"
            ) {
                trainingState =
                    Object.assign(
                        {},
                        trainingState,
                        message,
                        {
                            running:
                                false,

                            phase:
                                "stopped",

                            message:
                                "Training gestoppt."
                        }
                    );

                const worker =
                    trainingWorker;

                trainingWorker =
                    null;

                if (worker) {
                    try {
                        await worker.terminate();
                    } catch {}
                }

                return;
            }

            if (
                message.type ===
                "error"
            ) {
                trainingState =
                    Object.assign(
                        {},
                        trainingState,
                        {
                            running:
                                false,

                            phase:
                                "error",

                            message:
                                message.error,

                            error:
                                message.error,

                            stack:
                                message.stack
                        }
                    );

                const worker =
                    trainingWorker;

                trainingWorker =
                    null;

                if (worker) {
                    try {
                        await worker.terminate();
                    } catch {}
                }
            }
        }
    );

    trainingWorker.on(
        "error",
        error => {
            console.error(
                "[TRAIN WORKER ERROR]",
                error
            );

            trainingState = {
                ...trainingState,

                running:
                    false,

                phase:
                    "error",

                message:
                    error.message,

                error:
                    error.message
            };

            trainingWorker =
                null;
        }
    );

    trainingWorker.on(
        "exit",
        code => {
            console.log(
                "[TRAIN WORKER EXIT]",
                code
            );

            if (
                code !== 0 &&
                trainingState.running
            ) {
                trainingState = {
                    ...trainingState,

                    running:
                        false,

                    phase:
                        "error",

                    message:
                        "Training-Worker beendet mit Code " +
                        code
                };
            }

            trainingWorker =
                null;
        }
    );

    return {
        success: true,
        started: true,
        options
    };
}

async function stopTraining() {
    if (!trainingWorker) {
        return {
            success: false,
            message:
                "Kein Training läuft."
        };
    }

    trainingState.message =
        "Training wird gestoppt...";

    trainingWorker.postMessage({
        type:
            "stop"
    });

    return {
        success: true,
        stopping: true
    };
}


/* =========================================================
   GENERIERUNG
   ========================================================= */

function startGeneration(
    payload
) {
    if (generateWorker) {
        return {
            success: false,
            message:
                "Generation läuft bereits."
        };
    }

    if (!fileExists(
        GENERATE_WORKER
    )) {
        return {
            success: false,
            message:
                "generate-worker.js wurde nicht gefunden."
        };
    }

    const prompt =
        String(
            payload.prompt ||
            ""
        ).trim();

    if (!prompt) {
        return {
            success: false,
            message:
                "Kein Prompt angegeben."
        };
    }

    generationState = {
        running: true,
        text: "",
        error: null,
        startedAt:
            new Date().toISOString(),
        finishedAt: null
    };

    generateWorker =
        new Worker(
            GENERATE_WORKER,
            {
                workerData: {
                    root:
                        ROOT,

                    gelernt:
                        LEARNED_DIR,

                    prompt,

                    options:
                        payload || {}
                }
            }
        );

    generateWorker.on(
        "message",
        message => {
            if (!message) {
                return;
            }

            if (
                message.type ===
                "token"
            ) {
                generationState.text +=
                    String(
                        message.token ||
                        ""
                    );

                return;
            }

            if (
                message.type ===
                "complete"
            ) {
                generationState = {
                    ...generationState,

                    running:
                        false,

                    text:
                        message.text ||
                        generationState.text,

                    finishedAt:
                        new Date().toISOString()
                };

                generateWorker =
                    null;

                return;
            }

            if (
                message.type ===
                "error"
            ) {
                generationState = {
                    ...generationState,

                    running:
                        false,

                    error:
                        message.error,

                    finishedAt:
                        new Date().toISOString()
                };

                generateWorker =
                    null;
            }
        }
    );

    generateWorker.on(
        "error",
        error => {
            console.error(
                "[GENERATE WORKER ERROR]",
                error
            );

            generationState = {
                ...generationState,

                running:
                    false,

                error:
                    error.message,

                finishedAt:
                    new Date().toISOString()
            };

            generateWorker =
                null;
        }
    );

    generateWorker.on(
        "exit",
        code => {
            console.log(
                "[GENERATE WORKER EXIT]",
                code
            );

            if (
                code !== 0 &&
                generationState.running
            ) {
                generationState = {
                    ...generationState,

                    running:
                        false,

                    error:
                        "Generation-Worker beendet mit Code " +
                        code,

                    finishedAt:
                        new Date().toISOString()
                };
            }

            generateWorker =
                null;
        }
    );

    return {
        success: true,
        started: true
    };
}

function stopGeneration() {
    if (!generateWorker) {
        return {
            success: false,
            message:
                "Keine Generation läuft."
        };
    }

    generateWorker.postMessage({
        type:
            "stop"
    });

    return {
        success: true,
        stopping: true
    };
}


/* =========================================================
   HTTP SERVER
   ========================================================= */

const server =
    http.createServer(
        async (req, res) => {
            try {
                if (
                    req.method ===
                    "OPTIONS"
                ) {
                    res.writeHead(
                        204,
                        {
                            "Access-Control-Allow-Origin":
                                "*",

                            "Access-Control-Allow-Headers":
                                "Content-Type",

                            "Access-Control-Allow-Methods":
                                "GET,POST,OPTIONS"
                        }
                    );

                    res.end();

                    return;
                }

                const url =
                    new URL(
                        req.url,
                        "http://" +
                            (
                                req.headers.host ||
                                "localhost"
                            )
                    );

                const pathname =
                    url.pathname;

                /* -------------------------
                   API
                   ------------------------- */

                if (
                    pathname ===
                    "/api/status"
                ) {
                    json(
                        res,
                        200,
                        getStatus()
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/files"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            files:
                                getDataFiles(),

                            daten:
                                DATA_DIR,

                            gelernt:
                                LEARNED_DIR
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/data"
                ) {
                    const files =
                        getDataFiles();

                    const result =
                        [];

                    for (
                        const filename
                        of files
                    ) {
                        const file =
                            path.join(
                                DATA_DIR,
                                filename
                            );

                        const stat =
                            fs.statSync(
                                file
                            );

                        result.push({
                            name:
                                filename,

                            size:
                                stat.size,

                            modified:
                                stat.mtime
                        });
                    }

                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            files:
                                result
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/model"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            exists:
                                fileExists(
                                    MODEL_FILE
                                ),

                            model:
                                readJSON(
                                    MODEL_FILE
                                )
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/tokenizer"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            exists:
                                fileExists(
                                    TOKENIZER_FILE
                                ),

                            tokenizer:
                                readJSON(
                                    TOKENIZER_FILE
                                )
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/config"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            config:
                                readJSON(
                                    CONFIG_FILE
                                )
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/training-state"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            ...trainingState
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/generate/status"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            ...generationState
                        }
                    );

                    return;
                }

                if (
                    pathname ===
                    "/api/reload"
                ) {
                    json(
                        res,
                        200,
                        {
                            success:
                                true,

                            status:
                                getStatus()
                        }
                    );

                    return;
                }

                /* -------------------------
                   POST
                   ------------------------- */

                if (
                    req.method ===
                    "POST" &&
                    pathname ===
                    "/api/train/start"
                ) {
                    const body =
                        await readBody(
                            req
                        );

                    const result =
                        startTraining(
                            body
                        );

                    json(
                        res,
                        result.success
                            ? 200
                            : 409,
                        result
                    );

                    return;
                }

                if (
                    req.method ===
                    "POST" &&
                    pathname ===
                    "/api/train/stop"
                ) {
                    const result =
                        await stopTraining();

                    json(
                        res,
                        200,
                        result
                    );

                    return;
                }

                if (
                    req.method ===
                    "POST" &&
                    pathname ===
                    "/api/generate"
                ) {
                    const body =
                        await readBody(
                            req
                        );

                    const result =
                        startGeneration(
                            body
                        );

                    json(
                        res,
                        result.success
                            ? 200
                            : 409,
                        result
                    );

                    return;
                }

                if (
                    req.method ===
                    "POST" &&
                    pathname ===
                    "/api/generate/stop"
                ) {
                    const result =
                        stopGeneration();

                    json(
                        res,
                        200,
                        result
                    );

                    return;
                }

                /* -------------------------
                   STATIC
                   ------------------------- */

                let filePath;

                if (
                    pathname ===
                    "/" ||
                    pathname ===
                    "/index.html"
                ) {
                    filePath =
                        path.join(
                            ROOT,
                            "index.html"
                        );
                } else {
                    const safePath =
                        path.normalize(
                            pathname
                        ).replace(
                            /^(\.\.[/\\])+/, ""
                        );

                    filePath =
                        path.join(
                            ROOT,
                            safePath
                        );
                }

                if (
                    !filePath.startsWith(
                        ROOT
                    )
                ) {
                    text(
                        res,
                        403,
                        "Forbidden"
                    );

                    return;
                }

                if (
                    !fileExists(
                        filePath
                    )
                ) {
                    text(
                        res,
                        404,
                        "Nicht gefunden."
                    );

                    return;
                }

                const extension =
                    path.extname(
                        filePath
                    ).toLowerCase();

                const mime = {
                    ".html":
                        "text/html; charset=utf-8",

                    ".js":
                        "application/javascript; charset=utf-8",

                    ".css":
                        "text/css; charset=utf-8",

                    ".json":
                        "application/json; charset=utf-8",

                    ".txt":
                        "text/plain; charset=utf-8",

                    ".png":
                        "image/png",

                    ".jpg":
                        "image/jpeg",

                    ".jpeg":
                        "image/jpeg",

                    ".svg":
                        "image/svg+xml",

                    ".ico":
                        "image/x-icon"
                }[
                    extension
                ] ||
                "application/octet-stream";

                res.writeHead(
                    200,
                    {
                        "Content-Type":
                            mime,

                        "Cache-Control":
                            "no-cache"
                    }
                );

                fs.createReadStream(
                    filePath
                ).pipe(res);

                return;

            } catch (error) {
                console.error(
                    "[SERVER ERROR]",
                    error
                );

                sendError(
                    res,
                    500,
                    "Interner Serverfehler.",
                    error
                );
            }
        }
    );


/* =========================================================
   SERVER START
   ========================================================= */

server.listen(
    PORT,
    "0.0.0.0",
    () => {
        console.log(
            "======================================"
        );

        console.log(
            " LUMORA"
        );

        console.log(
            "======================================"
        );

        console.log(
            "Server:     http://localhost:" +
            PORT
        );

        console.log(
            "DATEN:      " +
            DATA_DIR
        );

        console.log(
            "GELERNT:    " +
            LEARNED_DIR
        );

        console.log(
            "Model:      " +
            (
                fileExists(
                    MODEL_FILE
                )
                    ? "vorhanden"
                    : "noch nicht vorhanden"
            )
        );

        console.log(
            "Tokenizer:  " +
            (
                fileExists(
                    TOKENIZER_FILE
                )
                    ? "vorhanden"
                    : "noch nicht vorhanden"
            )
        );

        console.log(
            "Daten:      " +
            getDataFiles().length +
            " Datei(en)"
        );

        console.log(
            "GitHub:     " +
            (
                GITHUB_TOKEN
                    ? "aktiv (" +
                      GITHUB_OWNER +
                      "/" +
                      GITHUB_REPO +
                      ")"
                    : "nicht konfiguriert"
            )
        );

        console.log(
            "API:"
        );

        console.log(
            "GET  /api/status"
        );

        console.log(
            "GET  /api/files"
        );

        console.log(
            "GET  /api/data"
        );

        console.log(
            "GET  /api/model"
        );

        console.log(
            "GET  /api/tokenizer"
        );

        console.log(
            "GET  /api/config"
        );

        console.log(
            "GET  /api/training-state"
        );

        console.log(
            "GET  /api/generate/status"
        );

        console.log(
            "POST /api/train/start"
        );

        console.log(
            "POST /api/train/stop"
        );

        console.log(
            "POST /api/generate"
        );

        console.log(
            "POST /api/generate/stop"
        );

        console.log(
            "======================================"
        );
    }
);

process.on(
    "SIGTERM",
    async () => {
        console.log(
            "SIGTERM erhalten."
        );

        if (trainingWorker) {
            try {
                trainingWorker.postMessage({
                    type:
                        "stop"
                });
            } catch {}
        }

        if (generateWorker) {
            try {
                generateWorker.postMessage({
                    type:
                        "stop"
                });
            } catch {}
        }

        server.close(
            () => {
                process.exit(
                    0
                );
            }
        );
    }
);

process.on(
    "SIGINT",
    async () => {
        console.log(
            "SIGINT erhalten."
        );

        if (trainingWorker) {
            try {
                trainingWorker.postMessage({
                    type:
                        "stop"
                });
            } catch {}
        }

        if (generateWorker) {
            try {
                generateWorker.postMessage({
                    type:
                        "stop"
                });
            } catch {}
        }

        server.close(
            () => {
                process.exit(
                    0
                );
            }
        );
    }
);

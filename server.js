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
    message: "Bereit.",
    github: null
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
    let body;

    try {
        body = JSON.stringify(
            data,
            null,
            2
        );
    } catch (error) {
        body = JSON.stringify({
            success: false,
            error: "Antwort konnte nicht als JSON erstellt werden.",
            details: error.message
        });
    }

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

    res.end(String(value));
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
            let finished = false;

            req.on(
                "data",
                chunk => {
                    if (finished) {
                        return;
                    }

                    body += chunk;

                    if (
                        body.length >
                        10 * 1024 * 1024
                    ) {
                        finished = true;

                        reject(
                            new Error(
                                "Request zu groß."
                            )
                        );

                        try {
                            req.destroy();
                        } catch {}
                    }
                }
            );

            req.on(
                "end",
                () => {
                    if (finished) {
                        return;
                    }

                    finished = true;

                    if (!body) {
                        resolve({});
                        return;
                    }

                    try {
                        resolve(
                            JSON.parse(body)
                        );
                    } catch (error) {
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
                error => {
                    if (finished) {
                        return;
                    }

                    finished = true;
                    reject(error);
                }
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

function readJSON(
    file,
    fallback = null
) {
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
                    : JSON.stringify(body);

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
                        "CubeGo",

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

function githubContentsPath(
    filePath
) {
    return (
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
        )
    );
}

async function getGitHubFile(
    filePath
) {
    return githubRequest(
        "GET",
        githubContentsPath(
            filePath
        )
    );
}

async function downloadGitHubFile(
    filePath
) {
    const result =
        await getGitHubFile(
            filePath
        );

    if (
        !result ||
        !result.content
    ) {
        throw new Error(
            "GitHub-Datei enthält keinen Inhalt: " +
            filePath
        );
    }

    const clean =
        result.content.replace(
            /\s/g,
            ""
        );

    return Buffer.from(
        clean,
        "base64"
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
            "CubeGo: gelernte Dateien aktualisiert",

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


/* =========================================================
   GITHUB -> RENDER
   ========================================================= */

async function restoreLearnedFromGitHub() {
    if (!GITHUB_TOKEN) {
        console.log(
            "[GITHUB] Kein GITHUB_TOKEN. Wiederherstellung übersprungen."
        );

        return {
            success: false,
            restored: [],
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

    const restored = [];
    const missing = [];
    const failed = [];

    console.log(
        "[GITHUB] Prüfe gespeichertes Modell..."
    );

    for (
        const filename of files
    ) {
        const local =
            path.join(
                LEARNED_DIR,
                filename
            );

        try {
            const buffer =
                await downloadGitHubFile(
                    "GELERNT/" +
                    filename
                );

            /*
             * GitHub ist hier die dauerhafte Speicherung.
             * Die Datei wird bei jedem Neustart aktualisiert,
             * damit Render nicht mit einem alten/fehlenden
             * lokalen Stand arbeitet.
             */

            fs.writeFileSync(
                local,
                buffer
            );

            restored.push(
                filename
            );

            console.log(
                "[GITHUB] Wiederhergestellt: " +
                filename
            );
        } catch (error) {
            if (
                String(
                    error.message
                ).includes("404")
            ) {
                missing.push(
                    filename
                );

                console.log(
                    "[GITHUB] Nicht vorhanden: " +
                    filename
                );
            } else {
                failed.push({
                    file:
                        filename,

                    error:
                        error.message
                });

                console.error(
                    "[GITHUB] Fehler bei " +
                    filename +
                    ": " +
                    error.message
                );
            }
        }
    }

    return {
        success:
            failed.length === 0,

        restored,

        missing,

        failed
    };
}


/* =========================================================
   RENDER -> GITHUB
   ========================================================= */

async function syncLearnedToGitHub() {
    if (!GITHUB_TOKEN) {
        return {
            success: false,
            uploaded: [],
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
    const failed = [];

    for (
        const filename of files
    ) {
        const local =
            path.join(
                LEARNED_DIR,
                filename
            );

        if (
            !fileExists(local)
        ) {
            continue;
        }

        try {
            const buffer =
                fs.readFileSync(
                    local
                );

            await uploadGitHubFile(
                "GELERNT/" +
                filename,

                buffer,

                "CubeGo: " +
                filename +
                " aktualisiert"
            );

            uploaded.push(
                filename
            );

            console.log(
                "[GITHUB] Hochgeladen: " +
                filename
            );
        } catch (error) {
            failed.push({
                file:
                    filename,

                error:
                    error.message
            });

            console.error(
                "[GITHUB] Upload-Fehler bei " +
                filename +
                ": " +
                error.message
            );
        }
    }

    return {
        success:
            failed.length === 0,

        uploaded,

        failed
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

        state:
            fileExists(
                STATE_FILE
            ),

        modelFile:
            MODEL_FILE,

        tokenizerFile:
            TOKENIZER_FILE,

        configFile:
            CONFIG_FILE
    };
}

function getStatus() {
    return {
        success: true,

        name:
            "CubeGo",

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
    payload =
        payload &&
        typeof payload === "object"
            ? payload
            : {};

    if (trainingWorker) {
        return {
            success: false,
            message:
                "Training läuft bereits."
        };
    }

    if (
        !fileExists(
            TRAIN_WORKER
        )
    ) {
        return {
            success: false,
            message:
                "train-worker.js wurde nicht gefunden."
        };
    }

    /*
     * KLEINES MODELL
     *
     * Diese Werte sind absichtlich deutlich
     * kleiner als die vorherigen Defaults.
     *
     * Dadurch sollte das Training auf Render
     * wesentlich schneller laufen.
     */

    const options = {
        epochs:
            Number(
                payload.epochs ??
                1
            ),

        sequenceLength:
            Number(
                payload.sequenceLength ??
                16
            ),

        learningRate:
            Number(
                payload.learningRate ??
                0.0003
            ),

        contextSize:
            Number(
                payload.contextSize ??
                16
            ),

        embeddingSize:
            Number(
                payload.embeddingSize ??
                32
            ),

        layers:
            Number(
                payload.layers ??
                1
            ),

        heads:
            Number(
                payload.heads ??
                1
            ),

        headSize:
            Number(
                payload.headSize ??
                32
            ),

        feedForwardSize:
            Number(
                payload.feedForwardSize ??
                64
            )
    };

    /*
     * Sicherheitsgrenzen gegen versehentlich
     * riesige Trainingsparameter.
     */

    options.epochs =
        Math.max(
            1,
            Math.min(
                options.epochs,
                100
            )
        );

    options.sequenceLength =
        Math.max(
            4,
            Math.min(
                options.sequenceLength,
                512
            )
        );

    options.contextSize =
        Math.max(
            4,
            Math.min(
                options.contextSize,
                512
            )
        );

    options.embeddingSize =
        Math.max(
            8,
            Math.min(
                options.embeddingSize,
                1024
            )
        );

    options.layers =
        Math.max(
            1,
            Math.min(
                options.layers,
                24
            )
        );

    options.heads =
        Math.max(
            1,
            Math.min(
                options.heads,
                24
            )
        );

    options.headSize =
        Math.max(
            8,
            Math.min(
                options.headSize,
                256
            )
        );

    options.feedForwardSize =
        Math.max(
            16,
            Math.min(
                options.feedForwardSize,
                4096
            )
        );

    trainingState = {
        running: true,

        phase:
            "starting",

        progress:
            0,

        epoch:
            0,

        step:
            0,

        loss:
            null,

        totalEpochs:
            options.epochs,

        message:
            "Training wird gestartet...",

        options,

        github:
            null
    };

    console.log(
        "[TRAIN] Starte Training mit:",
        options
    );

    try {
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
    } catch (error) {
        trainingWorker = null;

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

        return {
            success: false,
            message:
                "Training konnte nicht gestartet werden.",
            error:
                error.message
        };
    }

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
                                true,

                            phase:
                                "training"
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

                /*
                 * Erst sicherstellen, dass die Dateien
                 * vorhanden sind, danach zu GitHub sichern.
                 */

                try {
                    const github =
                        await syncLearnedToGitHub();

                    trainingState.github =
                        github;

                    if (
                        github.success
                    ) {
                        trainingState.message =
                            "Training abgeschlossen und Modell zu GitHub gesichert.";
                    } else {
                        trainingState.message =
                            "Training abgeschlossen, aber GitHub-Sicherung hatte Fehler.";
                    }
                } catch (error) {
                    trainingState.github = {
                        success:
                            false,

                        message:
                            error.message
                    };

                    trainingState.message =
                        "Training abgeschlossen, aber GitHub-Sicherung fehlgeschlagen.";
                }

                if (worker) {
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
                                message.error ||
                                "Unbekannter Trainingsfehler.",

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
        success:
            true,

        started:
            true,

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

    try {
        trainingWorker.postMessage({
            type:
                "stop"
        });
    } catch (error) {
        return {
            success: false,
            message:
                error.message
        };
    }

    return {
        success:
            true,

        stopping:
            true
    };
}


/* =========================================================
   GENERIERUNG
   ========================================================= */

function startGeneration(
    payload
) {
    payload =
        payload &&
        typeof payload === "object"
            ? payload
            : {};

    if (generateWorker) {
        return {
            success: false,
            message:
                "Generation läuft bereits."
        };
    }

    if (
        !fileExists(
            GENERATE_WORKER
        )
    ) {
        return {
            success: false,
            message:
                "generate-worker.js wurde nicht gefunden."
        };
    }

    if (
        !fileExists(
            MODEL_FILE
        )
    ) {
        return {
            success: false,
            message:
                "Noch kein trainiertes Modell vorhanden."
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
        running:
            true,

        text:
            "",

        error:
            null,

        startedAt:
            new Date().toISOString(),

        finishedAt:
            null
    };

    try {
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
                            payload
                    }
                }
            );
    } catch (error) {
        generateWorker = null;

        generationState = {
            ...generationState,

            running:
                false,

            error:
                error.message,

            finishedAt:
                new Date().toISOString()
        };

        return {
            success: false,
            message:
                "Generation konnte nicht gestartet werden.",
            error:
                error.message
        };
    }

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
        success:
            true,

        started:
            true
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

    try {
        generateWorker.postMessage({
            type:
                "stop"
        });
    } catch (error) {
        return {
            success: false,
            message:
                error.message
        };
    }

    return {
        success:
            true,

        stopping:
            true
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


                /* =================================================
                   GET API
                   ================================================= */

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


                /* =================================================
                   POST API
                   ================================================= */

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


                /* =================================================
                   STATIC FILES
                   ================================================= */

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
                            /^(\\.\\.[/\\])+/, ""
                        );

                    filePath =
                        path.join(
                            ROOT,
                            safePath
                        );
                }

                const rootResolved =
                    path.resolve(
                        ROOT
                    );

                const fileResolved =
                    path.resolve(
                        filePath
                    );

                if (
                    !(
                        fileResolved ===
                            rootResolved ||
                        fileResolved.startsWith(
                            rootResolved +
                            path.sep
                        )
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

                if (
                    !res.headersSent
                ) {
                    sendError(
                        res,
                        500,
                        "Interner Serverfehler.",
                        error
                    );
                } else {
                    try {
                        res.end();
                    } catch {}
                }
            }
        }
    );


/* =========================================================
   SERVER START
   ========================================================= */

server.listen(
    PORT,
    "0.0.0.0",
    async () => {
        console.log(
            "======================================"
        );

        console.log(
            " CubeGo"
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
            "--------------------------------------"
        );

        /*
         * WICHTIG:
         * Nach einem Render-Neustart ist GELERNT normalerweise
         * leer. Deshalb holen wir das gespeicherte Modell
         * jetzt automatisch aus GitHub zurück.
         */

        if (GITHUB_TOKEN) {
            try {
                const restored =
                    await restoreLearnedFromGitHub();

                console.log(
                    "[GITHUB] Wiederherstellung abgeschlossen:",
                    restored
                );
            } catch (error) {
                console.error(
                    "[GITHUB] Wiederherstellung fehlgeschlagen:",
                    error
                );
            }
        }

        console.log(
            "--------------------------------------"
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
            "Config:     " +
            (
                fileExists(
                    CONFIG_FILE
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
            "--------------------------------------"
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


/* =========================================================
   SAUBERES BEENDEN
   ========================================================= */

async function shutdown(
    signal
) {
    console.log(
        signal +
        " erhalten."
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

    setTimeout(
        () => {
            process.exit(
                0
            );
        },
        5000
    );
}

process.on(
    "SIGTERM",
    () => {
        shutdown(
            "SIGTERM"
        );
    }
);

process.on(
    "SIGINT",
    () => {
        shutdown(
            "SIGINT"
        );
    }
);

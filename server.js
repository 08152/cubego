// server.js
// CUBEGO – lokaler KI-Server
// Keine externen Bibliotheken erforderlich.

"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");
const { Worker } = require("worker_threads");

const ROOT = __dirname;

const DATEN_DIR = path.join(ROOT, "DATEN");
const GELERNT_DIR = path.join(ROOT, "GELERNT");

const MODEL_FILE = path.join(GELERNT_DIR, "model.json");
const TOKENIZER_FILE = path.join(GELERNT_DIR, "tokenizer.json");
const CONFIG_FILE = path.join(GELERNT_DIR, "config.json");
const STATE_FILE = path.join(GELERNT_DIR, "training-state.json");

const TRAIN_WORKER = path.join(ROOT, "train-worker.js");
const GENERATE_WORKER = path.join(ROOT, "generate-worker.js");

const PORT = Number(process.env.PORT) || 3000;

let trainingWorker = null;
let trainingStatus = {
    running: false,
    phase: "idle",
    epoch: 0,
    epochs: 0,
    step: 0,
    totalSteps: 0,
    loss: null,
    progress: 0,
    message: "Bereit"
};

const generationWorkers = new Set();


// ============================================================
// ORDNER
// ============================================================

function ensureDirectories() {
    fs.mkdirSync(DATEN_DIR, { recursive: true });
    fs.mkdirSync(GELERNT_DIR, { recursive: true });
}


// ============================================================
// JSON
// ============================================================

function readJSON(file, fallback = null) {
    try {
        if (!fs.existsSync(file)) return fallback;

        const text = fs.readFileSync(file, "utf8");

        if (!text.trim()) return fallback;

        return JSON.parse(text);
    } catch (error) {
        console.error("JSON-Fehler:", file, error.message);
        return fallback;
    }
}

function writeJSON(file, data) {
    const temp = file + ".tmp";

    fs.writeFileSync(
        temp,
        JSON.stringify(data, null, 2),
        "utf8"
    );

    fs.renameSync(temp, file);
}


// ============================================================
// STANDARD-DATEIEN
// ============================================================

function ensureLearnedFiles() {
    if (!fs.existsSync(CONFIG_FILE)) {
        writeJSON(CONFIG_FILE, {
            version: 1,
            model: {
                vocabSize: 8192,
                contextSize: 256,
                embeddingSize: 192,
                layers: 6,
                heads: 6,
                headSize: 32,
                feedForwardSize: 512
            },
            generation: {
                maxTokens: 160,
                temperature: 0.82,
                topK: 40,
                topP: 0.92,
                repetitionPenalty: 1.08
            }
        });
    }

    if (!fs.existsSync(STATE_FILE)) {
        writeJSON(STATE_FILE, {
            running: false,
            epoch: 0,
            epochs: 0,
            step: 0,
            totalSteps: 0,
            loss: null,
            progress: 0,
            message: "Noch nicht trainiert",
            updatedAt: new Date().toISOString()
        });
    }
}


// ============================================================
// DATEN
// ============================================================

function getDataFiles() {
    if (!fs.existsSync(DATEN_DIR)) return [];

    return fs
        .readdirSync(DATEN_DIR, { withFileTypes: true })
        .filter(entry => entry.isFile())
        .map(entry => entry.name)
        .filter(name => {
            const lower = name.toLowerCase();

            return (
                lower.endsWith(".json") ||
                lower.endsWith(".txt") ||
                lower.endsWith(".jsonl")
            );
        });
}

function getDataInfo() {
    const files = getDataFiles();

    return files.map(name => {
        const file = path.join(DATEN_DIR, name);

        let size = 0;

        try {
            size = fs.statSync(file).size;
        } catch {}

        return {
            name,
            size
        };
    });
}


// ============================================================
// SICHERER PFAD
// ============================================================

function safePath(requestPath) {
    let decoded;

    try {
        decoded = decodeURIComponent(requestPath);
    } catch {
        return null;
    }

    decoded = decoded.replace(/\0/g, "");

    if (decoded === "/") {
        return path.join(ROOT, "index.html");
    }

    const relative = decoded.replace(/^[/\\]+/, "");

    const absolute = path.resolve(ROOT, relative);
    const relativeToRoot = path.relative(ROOT, absolute);

    if (
        relativeToRoot.startsWith("..") ||
        path.isAbsolute(relativeToRoot)
    ) {
        return null;
    }

    return absolute;
}


// ============================================================
// MIME
// ============================================================

function getMimeType(file) {
    const ext = path.extname(file).toLowerCase();

    const types = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".ico": "image/x-icon"
    };

    return types[ext] || "application/octet-stream";
}


// ============================================================
// HTTP HELPERS
// ============================================================

function sendJSON(res, statusCode, data) {
    const body = JSON.stringify(data);

    res.writeHead(statusCode, {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*"
    });

    res.end(body);
}

function sendText(res, statusCode, text) {
    res.writeHead(statusCode, {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*"
    });

    res.end(text);
}

function readBody(req) {
    return new Promise((resolve, reject) => {
        let body = "";

        req.on("data", chunk => {
            body += chunk.toString("utf8");

            if (body.length > 10 * 1024 * 1024) {
                reject(new Error("Request zu groß."));
                req.destroy();
            }
        });

        req.on("end", () => {
            resolve(body);
        });

        req.on("error", reject);
    });
}

async function readJSONBody(req) {
    const body = await readBody(req);

    if (!body.trim()) {
        return {};
    }

    try {
        return JSON.parse(body);
    } catch {
        throw new Error("Ungültiges JSON.");
    }
}


// ============================================================
// TRAINING
// ============================================================

function startTraining(options = {}) {
    if (trainingWorker) {
        throw new Error("Training läuft bereits.");
    }

    if (!fs.existsSync(TRAIN_WORKER)) {
        throw new Error("train-worker.js wurde nicht gefunden.");
    }

    trainingStatus = {
        running: true,
        phase: "starting",
        epoch: 0,
        epochs: Number(options.epochs) || 10,
        step: 0,
        totalSteps: 0,
        loss: null,
        progress: 0,
        message: "Training wird gestartet"
    };

    trainingWorker = new Worker(TRAIN_WORKER, {
        workerData: {
            root: ROOT,
            daten: DATEN_DIR,
            gelernt: GELERNT_DIR,
            options
        }
    });

    trainingWorker.on("message", message => {
        if (!message || typeof message !== "object") return;

        if (message.type === "progress") {
            trainingStatus = {
                ...trainingStatus,
                ...message,
                running: true
            };

            writeTrainingState();
        }

        else if (message.type === "started") {
            trainingStatus = {
                ...trainingStatus,
                ...message,
                running: true,
                phase: "training"
            };

            writeTrainingState();
        }

        else if (message.type === "epoch") {
            trainingStatus = {
                ...trainingStatus,
                ...message,
                running: true,
                phase: "training"
            };

            writeTrainingState();
        }

        else if (message.type === "saved") {
            trainingStatus = {
                ...trainingStatus,
                ...message,
                running: true,
                phase: "saving"
            };

            writeTrainingState();
        }

        else if (message.type === "finished") {
            trainingStatus = {
                ...trainingStatus,
                ...message,
                running: false,
                phase: "finished",
                progress: 1,
                message: message.message || "Training abgeschlossen"
            };

            writeTrainingState();

            trainingWorker = null;
        }

        else if (message.type === "stopped") {
            trainingStatus = {
                ...trainingStatus,
                ...message,
                running: false,
                phase: "stopped",
                message: message.message || "Training gestoppt"
            };

            writeTrainingState();

            trainingWorker = null;
        }

        else if (message.type === "error") {
            trainingStatus = {
                ...trainingStatus,
                running: false,
                phase: "error",
                message: message.error || "Training fehlgeschlagen"
            };

            writeTrainingState();

            trainingWorker = null;

            console.error(
                "TRAINING WORKER:",
                message.error || "Unbekannter Fehler"
            );
        }
    });

    trainingWorker.on("error", error => {
        console.error("TRAINING WORKER ERROR:", error);

        trainingStatus = {
            ...trainingStatus,
            running: false,
            phase: "error",
            message: error.message
        };

        writeTrainingState();

        trainingWorker = null;
    });

    trainingWorker.on("exit", code => {
        if (trainingWorker) {
            if (code !== 0) {
                trainingStatus = {
                    ...trainingStatus,
                    running: false,
                    phase: "error",
                    message: `Training Worker beendet: ${code}`
                };

                writeTrainingState();
            }

            trainingWorker = null;
        }
    });
}

function stopTraining() {
    if (!trainingWorker) {
        return false;
    }

    try {
        trainingWorker.postMessage({
            type: "stop"
        });

        return true;
    } catch {
        return false;
    }
}

function writeTrainingState() {
    try {
        writeJSON(STATE_FILE, {
            ...trainingStatus,
            updatedAt: new Date().toISOString()
        });
    } catch (error) {
        console.error(
            "Training-State konnte nicht gespeichert werden:",
            error.message
        );
    }
}


// ============================================================
// GENERIERUNG
// ============================================================

function startGeneration(request, res) {
    if (!fs.existsSync(GENERATE_WORKER)) {
        sendJSON(res, 500, {
            ok: false,
            error: "generate-worker.js wurde nicht gefunden."
        });

        return;
    }

    const prompt =
        typeof request.prompt === "string"
            ? request.prompt.trim()
            : "";

    if (!prompt) {
        sendJSON(res, 400, {
            ok: false,
            error: "Kein Prompt angegeben."
        });

        return;
    }

    const config = readJSON(CONFIG_FILE, {});

    const generationConfig =
        config && config.generation
            ? config.generation
            : {};

    const options = {
        maxTokens:
            Number(request.maxTokens) ||
            Number(generationConfig.maxTokens) ||
            160,

        temperature:
            typeof request.temperature === "number"
                ? request.temperature
                : Number(generationConfig.temperature) || 0.82,

        topK:
            Number(request.topK) ||
            Number(generationConfig.topK) ||
            40,

        topP:
            typeof request.topP === "number"
                ? request.topP
                : Number(generationConfig.topP) || 0.92,

        repetitionPenalty:
            typeof request.repetitionPenalty === "number"
                ? request.repetitionPenalty
                : Number(generationConfig.repetitionPenalty) || 1.08,

        contextSize:
            Number(request.contextSize) ||
            Number(config?.model?.contextSize) ||
            256
    };

    const history =
        Array.isArray(request.history)
            ? request.history.slice(-30)
            : [];

    const systemPrompt =
        typeof request.systemPrompt === "string"
            ? request.systemPrompt
            : "";

    const worker = new Worker(GENERATE_WORKER, {
        workerData: {
            root: ROOT,
            gelernt: GELERNT_DIR,
            prompt,
            history,
            systemPrompt,
            options
        }
    });

    generationWorkers.add(worker);

    let finished = false;
    let answer = "";
    let tokenCount = 0;
    let promptTokens = 0;

    function finish(statusCode, data) {
        if (finished) return;

        finished = true;

        generationWorkers.delete(worker);

        sendJSON(res, statusCode, data);
    }

    worker.on("message", message => {
        if (!message || typeof message !== "object") {
            return;
        }

        if (message.type === "started") {
            return;
        }

        if (message.type === "token") {
            if (typeof message.token === "string") {
                answer += message.token;
            }

            tokenCount =
                Number(message.tokenCount) ||
                tokenCount + 1;

            if (Number.isFinite(message.promptTokens)) {
                promptTokens = message.promptTokens;
            }

            return;
        }

        if (message.type === "complete") {
            answer =
                typeof message.text === "string"
                    ? message.text
                    : answer;

            tokenCount =
                Number(message.tokenCount) ||
                tokenCount;

            promptTokens =
                Number(message.promptTokens) ||
                promptTokens;

            finish(200, {
                ok: true,
                answer,
                text: answer,
                tokenCount,
                promptTokens,
                model: message.model || "CUBEGO",
                finished: true
            });

            return;
        }

        if (message.type === "error") {
            finish(500, {
                ok: false,
                error:
                    message.error ||
                    "Generierungsfehler."
            });

            return;
        }
    });

    worker.on("error", error => {
        console.error(
            "GENERATE WORKER ERROR:",
            error
        );

        finish(500, {
            ok: false,
            error: error.message
        });
    });

    worker.on("exit", code => {
        generationWorkers.delete(worker);

        if (!finished && code !== 0) {
            finish(500, {
                ok: false,
                error:
                    "Generate-Worker wurde unerwartet beendet."
            });
        }
    });

    res.on("close", () => {
        if (!finished) {
            try {
                worker.postMessage({
                    type: "stop"
                });
            } catch {}
        }
    });
}


// ============================================================
// API
// ============================================================

async function handleAPI(req, res, pathname) {

    // --------------------------------------------------------
    // STATUS
    // --------------------------------------------------------

    if (req.method === "GET" && pathname === "/api/status") {
        sendJSON(res, 200, {
            ok: true,
            server: true,
            name: "CUBEGO",
            port: PORT,
            training: trainingStatus.running,
            trainingStatus,
            modelExists: fs.existsSync(MODEL_FILE),
            tokenizerExists: fs.existsSync(TOKENIZER_FILE),
            configExists: fs.existsSync(CONFIG_FILE),
            dataFiles: getDataFiles().length,
            generationWorkers: generationWorkers.size
        });

        return true;
    }


    // --------------------------------------------------------
    // DATEIEN
    // --------------------------------------------------------

    if (req.method === "GET" && pathname === "/api/files") {
        sendJSON(res, 200, {
            ok: true,
            folder: "DATEN",
            files: getDataInfo()
        });

        return true;
    }


    // --------------------------------------------------------
    // DATEN
    // --------------------------------------------------------

    if (req.method === "GET" && pathname === "/api/data") {
        const files = getDataFiles();

        const data = {};

        for (const name of files) {
            const file = path.join(DATEN_DIR, name);

            try {
                if (
                    name.toLowerCase().endsWith(".json") ||
                    name.toLowerCase().endsWith(".jsonl")
                ) {
                    data[name] = fs.readFileSync(
                        file,
                        "utf8"
                    );
                } else {
                    data[name] = fs.readFileSync(
                        file,
                        "utf8"
                    );
                }
            } catch {
                data[name] = "";
            }
        }

        sendJSON(res, 200, {
            ok: true,
            files: data
        });

        return true;
    }


    // --------------------------------------------------------
    // MODELL
    // --------------------------------------------------------

    if (req.method === "GET" && pathname === "/api/model") {
        if (!fs.existsSync(MODEL_FILE)) {
            sendJSON(res, 404, {
                ok: false,
                error: "Noch kein trainiertes Modell vorhanden."
            });

            return true;
        }

        try {
            const model = readJSON(MODEL_FILE);

            sendJSON(res, 200, {
                ok: true,
                model
            });
        } catch (error) {
            sendJSON(res, 500, {
                ok: false,
                error: error.message
            });
        }

        return true;
    }


    // --------------------------------------------------------
    // TOKENIZER
    // --------------------------------------------------------

    if (
        req.method === "GET" &&
        pathname === "/api/tokenizer"
    ) {
        if (!fs.existsSync(TOKENIZER_FILE)) {
            sendJSON(res, 404, {
                ok: false,
                error: "Noch kein Tokenizer vorhanden."
            });

            return true;
        }

        try {
            const tokenizer = readJSON(TOKENIZER_FILE);

            sendJSON(res, 200, {
                ok: true,
                tokenizer
            });
        } catch (error) {
            sendJSON(res, 500, {
                ok: false,
                error: error.message
            });
        }

        return true;
    }


    // --------------------------------------------------------
    // CONFIG
    // --------------------------------------------------------

    if (req.method === "GET" && pathname === "/api/config") {
        sendJSON(res, 200, {
            ok: true,
            config: readJSON(CONFIG_FILE, {})
        });

        return true;
    }


    // --------------------------------------------------------
    // TRAINING STATE
    // --------------------------------------------------------

    if (
        req.method === "GET" &&
        pathname === "/api/training-state"
    ) {
        sendJSON(res, 200, {
            ok: true,
            state: readJSON(STATE_FILE, trainingStatus),
            live: trainingStatus
        });

        return true;
    }


    // --------------------------------------------------------
    // RELOAD
    // --------------------------------------------------------

    if (req.method === "GET" && pathname === "/api/reload") {
        ensureLearnedFiles();

        sendJSON(res, 200, {
            ok: true,
            modelExists: fs.existsSync(MODEL_FILE),
            tokenizerExists: fs.existsSync(TOKENIZER_FILE),
            dataFiles: getDataFiles()
        });

        return true;
    }


    // --------------------------------------------------------
    // TRAIN START
    // --------------------------------------------------------

    if (
        req.method === "POST" &&
        pathname === "/api/train/start"
    ) {
        try {
            const body = await readJSONBody(req);

            startTraining(body || {});

            sendJSON(res, 200, {
                ok: true,
                message: "Training gestartet.",
                status: trainingStatus
            });
        } catch (error) {
            sendJSON(res, 500, {
                ok: false,
                error: error.message
            });
        }

        return true;
    }


    // --------------------------------------------------------
    // TRAIN STOP
    // --------------------------------------------------------

    if (
        req.method === "POST" &&
        pathname === "/api/train/stop"
    ) {
        const stopped = stopTraining();

        sendJSON(res, 200, {
            ok: stopped,
            message: stopped
                ? "Stop-Signal gesendet."
                : "Kein Training läuft.",
            status: trainingStatus
        });

        return true;
    }


    // --------------------------------------------------------
    // GENERATE
    // --------------------------------------------------------

    if (
        req.method === "POST" &&
        pathname === "/api/generate"
    ) {
        try {
            const body = await readJSONBody(req);

            startGeneration(body, res);
        } catch (error) {
            sendJSON(res, 400, {
                ok: false,
                error: error.message
            });
        }

        return true;
    }


    // --------------------------------------------------------
    // GENERATE STOP
    // --------------------------------------------------------

    if (
        req.method === "POST" &&
        pathname === "/api/generate/stop"
    ) {
        let stopped = 0;

        for (const worker of generationWorkers) {
            try {
                worker.postMessage({
                    type: "stop"
                });

                stopped++;
            } catch {}
        }

        sendJSON(res, 200, {
            ok: true,
            stopped
        });

        return true;
    }


    return false;
}


// ============================================================
// STATIC FILES
// ============================================================

function serveStatic(req, res, pathname) {
    const file = safePath(pathname);

    if (!file) {
        sendText(res, 403, "Forbidden");
        return;
    }

    if (!fs.existsSync(file)) {
        sendText(res, 404, "Not Found");
        return;
    }

    let stat;

    try {
        stat = fs.statSync(file);
    } catch {
        sendText(res, 404, "Not Found");
        return;
    }

    if (!stat.isFile()) {
        sendText(res, 404, "Not Found");
        return;
    }

    const mime = getMimeType(file);

    res.writeHead(200, {
        "Content-Type": mime,
        "Cache-Control":
            file.endsWith("index.html")
                ? "no-cache"
                : "public, max-age=3600"
    });

    fs.createReadStream(file).pipe(res);
}


// ============================================================
// SERVER
// ============================================================

ensureDirectories();
ensureLearnedFiles();

const server = http.createServer(async (req, res) => {
    try {
        const parsed = new URL(
            req.url,
            `http://${req.headers.host || "localhost"}`
        );

        const pathname = parsed.pathname;

        // CORS
        if (req.method === "OPTIONS") {
            res.writeHead(204, {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods":
                    "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers":
                    "Content-Type"
            });

            res.end();

            return;
        }

        // API
        if (pathname.startsWith("/api/")) {
            const handled = await handleAPI(
                req,
                res,
                pathname
            );

            if (handled) return;

            sendJSON(res, 404, {
                ok: false,
                error: "API-Endpunkt nicht gefunden."
            });

            return;
        }

        // Static
        serveStatic(req, res, pathname);

    } catch (error) {
        console.error("SERVER ERROR:", error);

        if (!res.headersSent) {
            sendJSON(res, 500, {
                ok: false,
                error: error.message
            });
        }
    }
});


// ============================================================
// START
// ============================================================

server.listen(PORT, "0.0.0.0", () => {
    console.log("");
    console.log("======================================");
    console.log(" CUBEGO");
    console.log("======================================");
    console.log("");
    console.log(`Server:     http://localhost:${PORT}`);
    console.log(`DATEN:      ${DATEN_DIR}`);
    console.log(`GELERNT:    ${GELERNT_DIR}`);
    console.log("");
    console.log(
        `Model:      ${fs.existsSync(MODEL_FILE) ? "vorhanden" : "noch nicht vorhanden"}`
    );
    console.log(
        `Tokenizer:  ${fs.existsSync(TOKENIZER_FILE) ? "vorhanden" : "noch nicht vorhanden"}`
    );
    console.log(
        `Daten:      ${getDataFiles().length} Datei(en)`
    );
    console.log("");
    console.log("API:");
    console.log("GET  /api/status");
    console.log("GET  /api/files");
    console.log("GET  /api/data");
    console.log("GET  /api/model");
    console.log("GET  /api/tokenizer");
    console.log("GET  /api/config");
    console.log("GET  /api/training-state");
    console.log("POST /api/train/start");
    console.log("POST /api/train/stop");
    console.log("POST /api/generate");
    console.log("POST /api/generate/stop");
    console.log("");
    console.log("======================================");
});


// ============================================================
// SAUBERES BEENDEN
// ============================================================

function shutdown() {
    console.log("\nCUBEGO wird beendet...");

    if (trainingWorker) {
        try {
            trainingWorker.postMessage({
                type: "stop"
            });
        } catch {}
    }

    for (const worker of generationWorkers) {
        try {
            worker.postMessage({
                type: "stop"
            });
        } catch {}
    }

    server.close(() => {
        process.exit(0);
    });

    setTimeout(() => {
        process.exit(0);
    }, 3000);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

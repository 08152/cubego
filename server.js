"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");
const { Worker } = require("worker_threads");

const PORT = Number(process.env.PORT || 10000);
const HOST = "0.0.0.0";

const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, "DATEN");
const LEARNED_DIR = path.join(ROOT, "GELERNT");

const INDEX_FILE = path.join(ROOT, "index.html");
const TRAIN_WORKER = path.join(ROOT, "train-worker.js");
const GENERATE_WORKER = path.join(ROOT, "generate-worker.js");

const MODEL_FILE = path.join(LEARNED_DIR, "model.json");
const TOKENIZER_FILE = path.join(LEARNED_DIR, "tokenizer.json");
const CONFIG_FILE = path.join(LEARNED_DIR, "config.json");
const TRAINING_STATE_FILE = path.join(LEARNED_DIR, "training-state.json");

const GITHUB_TOKEN = process.env.GITHUB_TOKEN || "";
const GITHUB_OWNER = process.env.GITHUB_OWNER || "08152";
const GITHUB_REPO = process.env.GITHUB_REPO || "cubego";
const GITHUB_BRANCH = process.env.GITHUB_BRANCH || "main";

let trainingWorker = null;
let generateWorker = null;

let trainingState = {
    running: false,
    progress: 0,
    status: "Noch nicht trainiert",
    error: null,
    startedAt: null,
    finishedAt: null
};

let generationState = {
    running: false
};

function ensureDirectories() {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.mkdirSync(LEARNED_DIR, { recursive: true });
}

ensureDirectories();

function safeReadJSON(file, fallback = null) {
    try {
        if (!fs.existsSync(file)) return fallback;
        return JSON.parse(fs.readFileSync(file, "utf8"));
    } catch {
        return fallback;
    }
}

function listFilesRecursive(dir) {
    if (!fs.existsSync(dir)) return [];

    const result = [];

    function walk(current) {
        for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
            const full = path.join(current, entry.name);

            if (entry.isDirectory()) {
                walk(full);
            } else {
                result.push(path.relative(ROOT, full).replace(/\\/g, "/"));
            }
        }
    }

    walk(dir);
    return result;
}

function sendJSON(res, status, data) {
    const body = JSON.stringify(data, null, 2);

    res.writeHead(status, {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
    });

    res.end(body);
}

function sendText(res, status, text, contentType = "text/plain; charset=utf-8") {
    res.writeHead(status, {
        "Content-Type": contentType,
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*"
    });

    res.end(text);
}

function parseBody(req) {
    return new Promise((resolve, reject) => {
        let body = "";

        req.on("data", chunk => {
            body += chunk;

            if (body.length > 20 * 1024 * 1024) {
                reject(new Error("Request zu groß."));
                req.destroy();
            }
        });

        req.on("end", () => {
            if (!body) {
                resolve({});
                return;
            }

            try {
                resolve(JSON.parse(body));
            } catch {
                reject(new Error("Ungültiges JSON."));
            }
        });

        req.on("error", reject);
    });
}

function modelExists() {
    return fs.existsSync(MODEL_FILE);
}

function tokenizerExists() {
    return fs.existsSync(TOKENIZER_FILE);
}

function trainingFilesExist() {
    return modelExists() && tokenizerExists();
}

function getStatus() {
    return {
        ok: true,
        name: "LUMORA",
        server: true,
        training: {
            running: trainingState.running,
            progress: trainingState.progress,
            status: trainingState.status,
            error: trainingState.error,
            startedAt: trainingState.startedAt,
            finishedAt: trainingState.finishedAt
        },
        generation: {
            running: generationState.running
        },
        data: {
            directory: DATA_DIR,
            files: fs.existsSync(DATA_DIR)
                ? fs.readdirSync(DATA_DIR)
                : [],
            count: fs.existsSync(DATA_DIR)
                ? fs.readdirSync(DATA_DIR).length
                : 0
        },
        learned: {
            directory: LEARNED_DIR,
            exists: fs.existsSync(LEARNED_DIR),
            model: modelExists(),
            tokenizer: tokenizerExists(),
            config: fs.existsSync(CONFIG_FILE),
            trainingState: fs.existsSync(TRAINING_STATE_FILE)
        },
        github: {
            enabled: Boolean(GITHUB_TOKEN),
            owner: GITHUB_OWNER,
            repository: GITHUB_REPO,
            branch: GITHUB_BRANCH
        }
    };
}

/* =========================================================
   GITHUB
   ========================================================= */

function githubRequest(method, apiPath, body = null) {
    return new Promise((resolve, reject) => {
        if (!GITHUB_TOKEN) {
            reject(new Error("GITHUB_TOKEN ist nicht gesetzt."));
            return;
        }

        const https = require("https");

        const options = {
            hostname: "api.github.com",
            path: apiPath,
            method,
            headers: {
                "User-Agent": "LUMORA-Render",
                "Authorization": `Bearer ${GITHUB_TOKEN}`,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
        };

        let payload = null;

        if (body !== null) {
            payload = JSON.stringify(body);

            options.headers["Content-Type"] = "application/json";
            options.headers["Content-Length"] =
                Buffer.byteLength(payload);
        }

        const request = https.request(options, response => {
            let data = "";

            response.setEncoding("utf8");

            response.on("data", chunk => {
                data += chunk;
            });

            response.on("end", () => {
                let parsed = null;

                try {
                    parsed = data ? JSON.parse(data) : null;
                } catch {
                    parsed = data;
                }

                if (response.statusCode >= 200 && response.statusCode < 300) {
                    resolve(parsed);
                    return;
                }

                const message =
                    parsed &&
                    typeof parsed === "object" &&
                    parsed.message
                        ? parsed.message
                        : `GitHub HTTP ${response.statusCode}`;

                reject(
                    new Error(
                        `GitHub API Fehler ${response.statusCode}: ${message}`
                    )
                );
            });
        });

        request.on("error", reject);

        if (payload) {
            request.write(payload);
        }

        request.end();
    });
}

function githubPath(filePath) {
    return filePath
        .replace(/\\/g, "/")
        .replace(/^\/+/, "");
}

async function getGithubFile(filePath) {
    const encoded = githubPath(filePath)
        .split("/")
        .map(encodeURIComponent)
        .join("/");

    try {
        return await githubRequest(
            "GET",
            `/repos/${encodeURIComponent(GITHUB_OWNER)}/${encodeURIComponent(
                GITHUB_REPO
            )}/contents/${encoded}?ref=${encodeURIComponent(GITHUB_BRANCH)}`
        );
    } catch (error) {
        if (
            String(error.message).includes("404") ||
            String(error.message).toLowerCase().includes("not found")
        ) {
            return null;
        }

        throw error;
    }
}

async function uploadFileToGitHub(relativePath) {
    if (!GITHUB_TOKEN) {
        throw new Error(
            "GITHUB_TOKEN fehlt. GitHub-Synchronisierung wurde nicht aktiviert."
        );
    }

    const absolutePath = path.join(ROOT, relativePath);

    if (!fs.existsSync(absolutePath)) {
        throw new Error(
            `Datei für GitHub nicht gefunden: ${relativePath}`
        );
    }

    const content = fs.readFileSync(absolutePath);
    const base64 = content.toString("base64");

    const existing = await getGithubFile(relativePath);

    const payload = {
        message: `LUMORA: Update ${relativePath}`,
        content: base64,
        branch: GITHUB_BRANCH
    };

    if (existing && existing.sha) {
        payload.sha = existing.sha;
    }

    return await githubRequest(
        "PUT",
        `/repos/${encodeURIComponent(GITHUB_OWNER)}/${encodeURIComponent(
            GITHUB_REPO
        )}/contents/${githubPath(relativePath)}`,
        payload
    );
}

async function syncLearnedToGitHub() {
    if (!GITHUB_TOKEN) {
        console.log(
            "[GitHub] GITHUB_TOKEN nicht gesetzt – keine Synchronisierung."
        );

        return {
            enabled: false,
            uploaded: []
        };
    }

    const files = [
        "GELERNT/model.json",
        "GELERNT/tokenizer.json",
        "GELERNT/config.json",
        "GELERNT/training-state.json"
    ];

    const uploaded = [];

    for (const file of files) {
        if (!fs.existsSync(path.join(ROOT, file))) {
            console.log(`[GitHub] Überspringe ${file} – nicht vorhanden.`);
            continue;
        }

        console.log(`[GitHub] Lade ${file} hoch...`);

        await uploadFileToGitHub(file);

        uploaded.push(file);

        console.log(`[GitHub] ${file} gespeichert.`);
    }

    return {
        enabled: true,
        uploaded
    };
}

/* =========================================================
   TRAINING
   ========================================================= */

function startTraining() {
    if (trainingWorker) {
        throw new Error("Training läuft bereits.");
    }

    if (!fs.existsSync(TRAIN_WORKER)) {
        throw new Error("train-worker.js wurde nicht gefunden.");
    }

    ensureDirectories();

    trainingState = {
        running: true,
        progress: 0,
        status: "Training wird gestartet...",
        error: null,
        startedAt: new Date().toISOString(),
        finishedAt: null
    };

    trainingWorker = new Worker(TRAIN_WORKER);

    trainingWorker.on("message", async message => {
        try {
            if (!message || typeof message !== "object") {
                return;
            }

            if (message.type === "progress") {
                trainingState.progress =
                    Number(message.progress ?? message.percent ?? 0);

                trainingState.status =
                    message.status ||
                    message.message ||
                    "Training läuft...";

                return;
            }

            if (
                message.type === "status" ||
                message.type === "log"
            ) {
                trainingState.status =
                    message.status ||
                    message.message ||
                    trainingState.status;

                console.log(
                    "[TRAIN]",
                    message.message ||
                    message.status ||
                    ""
                );

                return;
            }

            if (message.type === "error") {
                trainingState.running = false;
                trainingState.error =
                    message.error ||
                    message.message ||
                    "Unbekannter Trainingsfehler.";

                trainingState.status = "Training fehlgeschlagen.";
                trainingState.finishedAt =
                    new Date().toISOString();

                console.error(
                    "[TRAIN ERROR]",
                    trainingState.error
                );

                if (trainingWorker) {
                    await trainingWorker.terminate();
                    trainingWorker = null;
                }

                return;
            }

            if (
                message.type === "complete" ||
                message.type === "done" ||
                message.type === "finished"
            ) {
                await finishTraining();
            }
        } catch (error) {
            console.error(
                "[TRAIN MESSAGE ERROR]",
                error
            );
        }
    });

    trainingWorker.on("error", error => {
        console.error("[TRAIN WORKER ERROR]", error);

        trainingState.running = false;
        trainingState.status = "Training fehlgeschlagen.";
        trainingState.error = error.message;
        trainingState.finishedAt =
            new Date().toISOString();

        trainingWorker = null;
    });

    trainingWorker.on("exit", async code => {
        console.log("[TRAIN WORKER EXIT]", code);

        if (
            trainingState.running &&
            code !== 0
        ) {
            trainingState.running = false;
            trainingState.status = "Training beendet.";
            trainingState.error =
                `Training-Worker beendet mit Code ${code}.`;
            trainingState.finishedAt =
                new Date().toISOString();
        }

        trainingWorker = null;
    });

    trainingWorker.postMessage({
        type: "start",
        action: "train"
    });
}

async function finishTraining() {
    console.log("[TRAIN] Training erfolgreich beendet.");

    ensureDirectories();

    const model = modelExists();
    const tokenizer = tokenizerExists();

    console.log(
        `[TRAIN] model.json: ${model ? "OK" : "FEHLT"}`
    );

    console.log(
        `[TRAIN] tokenizer.json: ${tokenizer ? "OK" : "FEHLT"}`
    );

    if (!model || !tokenizer) {
        trainingState.running = false;
        trainingState.progress = 100;
        trainingState.status =
            "Training beendet, aber Model/Tokenizer fehlen.";
        trainingState.error =
            `Erwartet wurden ${!model ? "GELERNT/model.json" : ""}` +
            `${!model && !tokenizer ? " und " : ""}` +
            `${!tokenizer ? "GELERNT/tokenizer.json" : ""}.`;
        trainingState.finishedAt =
            new Date().toISOString();

        console.error(
            "[TRAIN]",
            trainingState.error
        );

        return;
    }

    trainingState.running = false;
    trainingState.progress = 100;
    trainingState.status =
        "Training erfolgreich abgeschlossen.";
    trainingState.error = null;
    trainingState.finishedAt =
        new Date().toISOString();

    fs.writeFileSync(
        TRAINING_STATE_FILE,
        JSON.stringify(trainingState, null, 2),
        "utf8"
    );

    try {
        const result = await syncLearnedToGitHub();

        console.log(
            "[GitHub] Synchronisierung abgeschlossen:",
            result.uploaded
        );

        trainingState.status =
            "Training fertig – Modell in GitHub gespeichert.";

        fs.writeFileSync(
            TRAINING_STATE_FILE,
            JSON.stringify(trainingState, null, 2),
            "utf8"
        );

        /*
         * training-state.json wurde gerade nach GitHub geschrieben.
         * Wenn sich der Inhalt danach geändert hat, wird es noch einmal
         * hochgeladen.
         */
        if (GITHUB_TOKEN) {
            try {
                await uploadFileToGitHub(
                    "GELERNT/training-state.json"
                );
            } catch (error) {
                console.error(
                    "[GitHub] training-state.json konnte nicht aktualisiert werden:",
                    error.message
                );
            }
        }
    } catch (error) {
        console.error(
            "[GitHub ERROR]",
            error.message
        );

        trainingState.status =
            "Training fertig – GitHub-Speicherung fehlgeschlagen.";
        trainingState.error =
            `GitHub: ${error.message}`;

        fs.writeFileSync(
            TRAINING_STATE_FILE,
            JSON.stringify(trainingState, null, 2),
            "utf8"
        );
    }
}

function stopTraining() {
    if (!trainingWorker) {
        return false;
    }

    try {
        trainingWorker.postMessage({
            type: "stop",
            action: "stop"
        });
    } catch {}

    return true;
}

/* =========================================================
   GENERATION
   ========================================================= */

function startGeneration(payload) {
    if (generateWorker) {
        throw new Error("Generierung läuft bereits.");
    }

    if (!fs.existsSync(GENERATE_WORKER)) {
        throw new Error("generate-worker.js wurde nicht gefunden.");
    }

    if (!trainingFilesExist()) {
        throw new Error(
            "Die KI wurde noch nicht vollständig trainiert. " +
            "GELERNT/model.json und GELERNT/tokenizer.json fehlen."
        );
    }

    generationState.running = true;

    generateWorker = new Worker(GENERATE_WORKER);

    generateWorker.on("message", message => {
        if (!message || typeof message !== "object") {
            return;
        }

        if (message.type === "complete") {
            generationState.running = false;
        }

        if (message.type === "error") {
            generationState.running = false;
        }
    });

    generateWorker.on("error", error => {
        console.error(
            "[GENERATE WORKER ERROR]",
            error
        );

        generationState.running = false;
    });

    generateWorker.on("exit", () => {
        generateWorker = null;
        generationState.running = false;
    });

    generateWorker.postMessage({
        type: "generate",
        action: "generate",
        ...payload
    });

    return true;
}

function stopGeneration() {
    if (!generateWorker) {
        return false;
    }

    try {
        generateWorker.postMessage({
            type: "stop",
            action: "stop"
        });
    } catch {}

    return true;
}

/* =========================================================
   STATIC FILES
   ========================================================= */

const MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon"
};

function serveStatic(req, res, pathname) {
    let requested = pathname;

    if (requested === "/") {
        requested = "/index.html";
    }

    requested = decodeURIComponent(requested);

    const filePath = path.normalize(
        path.join(ROOT, requested)
    );

    if (
        !filePath.startsWith(ROOT) ||
        !fs.existsSync(filePath) ||
        !fs.statSync(filePath).isFile()
    ) {
        return false;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType =
        MIME[ext] ||
        "application/octet-stream";

    res.writeHead(200, {
        "Content-Type": contentType,
        "Cache-Control": "no-cache"
    });

    fs.createReadStream(filePath).pipe(res);

    return true;
}

/* =========================================================
   SERVER
   ========================================================= */

const server = http.createServer(async (req, res) => {
    try {
        if (req.method === "OPTIONS") {
            res.writeHead(204, {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
            });

            res.end();
            return;
        }

        const url = new URL(
            req.url,
            `http://${req.headers.host || "localhost"}`
        );

        const pathname = url.pathname;

        /* STATUS */

        if (
            req.method === "GET" &&
            pathname === "/api/status"
        ) {
            sendJSON(res, 200, getStatus());
            return;
        }

        /* FILES */

        if (
            req.method === "GET" &&
            pathname === "/api/files"
        ) {
            sendJSON(res, 200, {
                data: listFilesRecursive(DATA_DIR),
                learned: listFilesRecursive(LEARNED_DIR)
            });

            return;
        }

        /* DATA */

        if (
            req.method === "GET" &&
            pathname === "/api/data"
        ) {
            const files = listFilesRecursive(DATA_DIR);

            const data = [];

            for (const relative of files) {
                const full = path.join(ROOT, relative);

                if (relative.endsWith(".json")) {
                    data.push({
                        file: relative,
                        content: safeReadJSON(full)
                    });
                } else {
                    data.push({
                        file: relative,
                        content: fs.readFileSync(
                            full,
                            "utf8"
                        )
                    });
                }
            }

            sendJSON(res, 200, data);
            return;
        }

        /* MODEL */

        if (
            req.method === "GET" &&
            pathname === "/api/model"
        ) {
            if (!modelExists()) {
                sendJSON(res, 404, {
                    exists: false,
                    error: "GELERNT/model.json wurde noch nicht erstellt."
                });

                return;
            }

            sendJSON(res, 200, {
                exists: true,
                model: safeReadJSON(MODEL_FILE)
            });

            return;
        }

        /* TOKENIZER */

        if (
            req.method === "GET" &&
            pathname === "/api/tokenizer"
        ) {
            if (!tokenizerExists()) {
                sendJSON(res, 404, {
                    exists: false,
                    error:
                        "GELERNT/tokenizer.json wurde noch nicht erstellt."
                });

                return;
            }

            sendJSON(res, 200, {
                exists: true,
                tokenizer:
                    safeReadJSON(TOKENIZER_FILE)
            });

            return;
        }

        /* CONFIG */

        if (
            req.method === "GET" &&
            pathname === "/api/config"
        ) {
            sendJSON(res, 200, {
                exists: fs.existsSync(CONFIG_FILE),
                config: safeReadJSON(CONFIG_FILE, {})
            });

            return;
        }

        /* TRAINING STATE */

        if (
            req.method === "GET" &&
            pathname === "/api/training-state"
        ) {
            sendJSON(res, 200, {
                ...trainingState,
                file:
                    safeReadJSON(
                        TRAINING_STATE_FILE,
                        null
                    )
            });

            return;
        }

        /* RELOAD */

        if (
            req.method === "GET" &&
            pathname === "/api/reload"
        ) {
            ensureDirectories();

            sendJSON(res, 200, getStatus());
            return;
        }

        /* START TRAINING */

        if (
            req.method === "POST" &&
            pathname === "/api/train/start"
        ) {
            if (trainingWorker) {
                sendJSON(res, 409, {
                    ok: false,
                    error: "Training läuft bereits."
                });

                return;
            }

            startTraining();

            sendJSON(res, 200, {
                ok: true,
                message: "Training gestartet.",
                status: getStatus()
            });

            return;
        }

        /* STOP TRAINING */

        if (
            req.method === "POST" &&
            pathname === "/api/train/stop"
        ) {
            const stopped = stopTraining();

            sendJSON(res, 200, {
                ok: stopped,
                message: stopped
                    ? "Training wird gestoppt."
                    : "Kein Training läuft."
            });

            return;
        }

        /* START GENERATION */

        if (
            req.method === "POST" &&
            pathname === "/api/generate"
        ) {
            const payload = await parseBody(req);

            try {
                startGeneration(payload);

                sendJSON(res, 200, {
                    ok: true,
                    message: "Generierung gestartet."
                });
            } catch (error) {
                sendJSON(res, 500, {
                    ok: false,
                    error: error.message
                });
            }

            return;
        }

        /* STOP GENERATION */

        if (
            req.method === "POST" &&
            pathname === "/api/generate/stop"
        ) {
            const stopped = stopGeneration();

            sendJSON(res, 200, {
                ok: stopped,
                message: stopped
                    ? "Generierung wird gestoppt."
                    : "Keine Generierung läuft."
            });

            return;
        }

        /* STATIC */

        if (req.method === "GET") {
            if (serveStatic(req, res, pathname)) {
                return;
            }
        }

        sendJSON(res, 404, {
            ok: false,
            error: "Nicht gefunden."
        });
    } catch (error) {
        console.error("[SERVER ERROR]", error);

        sendJSON(res, 500, {
            ok: false,
            error: error.message || "Interner Serverfehler."
        });
    }
});

/* =========================================================
   START
   ========================================================= */

server.listen(PORT, HOST, () => {
    console.log("======================================");
    console.log(" LUMORA");
    console.log("======================================");
    console.log(`Server:     http://localhost:${PORT}`);
    console.log(`DATEN:      ${DATA_DIR}`);
    console.log(`GELERNT:    ${LEARNED_DIR}`);
    console.log(
        `Model:      ${modelExists() ? "vorhanden" : "noch nicht vorhanden"}`
    );
    console.log(
        `Tokenizer:  ${
            tokenizerExists()
                ? "vorhanden"
                : "noch nicht vorhanden"
        }`
    );
    console.log(
        `Daten:      ${listFilesRecursive(DATA_DIR).length} Datei(en)`
    );
    console.log(
        `GitHub:     ${
            GITHUB_TOKEN
                ? `aktiv (${GITHUB_OWNER}/${GITHUB_REPO})`
                : "nicht aktiviert"
        }`
    );
    console.log("API:");
    console.log("GET  /api/status");
    console.log("GET  /api/files");
    console.log("GET  /api/data");
    console.log("GET  /api/model");
    console.log("GET  /api/tokenizer");
    console.log("GET  /api/config");
    console.log("GET  /api/training-state");
    console.log("GET  /api/reload");
    console.log("POST /api/train/start");
    console.log("POST /api/train/stop");
    console.log("POST /api/generate");
    console.log("POST /api/generate/stop");
    console.log("======================================");
});

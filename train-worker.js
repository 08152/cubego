"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const https = require("https");
const { Worker } = require("worker_threads");

const ROOT = __dirname;

const DATA_DIR = path.join(ROOT, "DATEN");
const LEARNED_DIR = path.join(ROOT, "GELERNT");

const MODEL_FILE = path.join(
    LEARNED_DIR,
    "model.json"
);

const TOKENIZER_FILE = path.join(
    LEARNED_DIR,
    "tokenizer.json"
);

const CONFIG_FILE = path.join(
    LEARNED_DIR,
    "config.json"
);

const TRAINING_STATE_FILE = path.join(
    LEARNED_DIR,
    "training-state.json"
);

const PORT =
    Number(process.env.PORT) || 10000;

const GITHUB_TOKEN =
    process.env.GITHUB_TOKEN || "";

const GITHUB_OWNER =
    process.env.GITHUB_OWNER || "08152";

const GITHUB_REPO =
    process.env.GITHUB_REPO || "cubego";

const GITHUB_BRANCH =
    process.env.GITHUB_BRANCH || "main";

const TRAIN_WORKER =
    path.join(
        ROOT,
        "train-worker.js"
    );

const GENERATE_WORKER =
    path.join(
        ROOT,
        "generate-worker.js"
    );

fs.mkdirSync(
    DATA_DIR,
    {
        recursive: true
    }
);

fs.mkdirSync(
    LEARNED_DIR,
    {
        recursive: true
    }
);

let trainingWorker = null;
let generateWorker = null;

let trainingState = {
    running: false,
    phase: "idle",
    progress: 0,
    epoch: 0,
    step: 0,
    loss: null,
    message: "Noch nicht trainiert.",
    startedAt: null,
    finishedAt: null,
    error: null
};

let generationState = {
    running: false,
    text: "",
    error: null
};


/* =========================================================
   HILFSFUNKTIONEN
   ========================================================= */

function jsonResponse(
    res,
    status,
    data
) {
    const body =
        JSON.stringify(
            data,
            null,
            2
        );

    res.writeHead(
        status,
        {
            "Content-Type":
                "application/json; charset=utf-8",

            "Access-Control-Allow-Origin":
                "*",

            "Access-Control-Allow-Methods":
                "GET,POST,OPTIONS",

            "Access-Control-Allow-Headers":
                "Content-Type"
        }
    );

    res.end(body);
}

function textResponse(
    res,
    status,
    text,
    contentType =
        "text/plain; charset=utf-8"
) {
    res.writeHead(
        status,
        {
            "Content-Type":
                contentType,

            "Access-Control-Allow-Origin":
                "*"
        }
    );

    res.end(text);
}

function readJSON(file) {
    return JSON.parse(
        fs.readFileSync(
            file,
            "utf8"
        )
    );
}

function safeReadJSON(file) {
    try {
        if (
            !fs.existsSync(file)
        ) {
            return null;
        }

        return readJSON(file);
    } catch {
        return null;
    }
}

function readRequestBody(req) {
    return new Promise(
        (resolve, reject) => {
            let body = "";

            req.on(
                "data",
                chunk => {
                    body += chunk;

                    if (
                        body.length >
                        20 * 1024 * 1024
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
                            JSON.parse(body)
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
                        "GITHUB_TOKEN ist nicht gesetzt."
                    )
                );

                return;
            }

            const requestBody =
                body === null
                    ? null
                    : JSON.stringify(body);

            const options = {
                hostname:
                    "api.github.com",

                path:
                    apiPath,

                method:
                    method,

                headers: {
                    "User-Agent":
                        "LUMORA-Render",

                    "Accept":
                        "application/vnd.github+json",

                    "X-GitHub-Api-Version":
                        "2022-11-28",

                    "Authorization":
                        "Bearer " +
                        GITHUB_TOKEN
                }
            };

            if (requestBody) {
                options.headers[
                    "Content-Type"
                ] =
                    "application/json";

                options.headers[
                    "Content-Length"
                ] =
                    Buffer.byteLength(
                        requestBody
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
                                data +=
                                    chunk;
                            }
                        );

                        response.on(
                            "end",
                            () => {
                                let parsed =
                                    null;

                                try {
                                    parsed =
                                        data
                                            ? JSON.parse(
                                                  data
                                              )
                                            : null;
                                } catch {
                                    parsed =
                                        data;
                                }

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
                                    const message =
                                        parsed &&
                                        parsed.message
                                            ? parsed.message
                                            : "GitHub API Fehler";

                                    reject(
                                        new Error(
                                            "GitHub " +
                                                response.statusCode +
                                                ": " +
                                                message
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

            if (requestBody) {
                request.write(
                    requestBody
                );
            }

            request.end();
        }
    );
}

async function getGithubFile(
    repoPath
) {
    const apiPath =
        `/repos/${encodeURIComponent(
            GITHUB_OWNER
        )}/${encodeURIComponent(
            GITHUB_REPO
        )}/contents/${repoPath}?ref=${encodeURIComponent(
            GITHUB_BRANCH
        )}`;

    try {
        return await githubRequest(
            "GET",
            apiPath
        );
    } catch (error) {
        if (
            error.message.includes(
                "GitHub 404"
            )
        ) {
            return null;
        }

        throw error;
    }
}

async function uploadFileToGitHub(
    localFile,
    repoPath,
    message
) {
    if (!GITHUB_TOKEN) {
        throw new Error(
            "GITHUB_TOKEN ist nicht gesetzt."
        );
    }

    if (
        !fileExists(localFile)
    ) {
        throw new Error(
            "Datei existiert nicht: " +
            localFile
        );
    }

    const content =
        fs.readFileSync(
            localFile
        );

    const existing =
        await getGithubFile(
            repoPath
        );

    const payload = {
        message:
            message ||
            "LUMORA: gelernte Dateien aktualisieren",

        content:
            content.toString(
                "base64"
            ),

        branch:
            GITHUB_BRANCH
    };

    if (
        existing &&
        existing.sha
    ) {
        payload.sha =
            existing.sha;
    }

    const apiPath =
        `/repos/${encodeURIComponent(
            GITHUB_OWNER
        )}/${encodeURIComponent(
            GITHUB_REPO
        )}/contents/${repoPath}`;

    return await githubRequest(
        "PUT",
        apiPath,
        payload
    );
}

async function syncLearnedToGitHub() {
    if (!GITHUB_TOKEN) {
        return {
            enabled: false,
            message:
                "GITHUB_TOKEN nicht gesetzt."
        };
    }

    const files = [
        {
            local:
                MODEL_FILE,

            remote:
                "GELERNT/model.json"
        },

        {
            local:
                TOKENIZER_FILE,

            remote:
                "GELERNT/tokenizer.json"
        },

        {
            local:
                CONFIG_FILE,

            remote:
                "GELERNT/config.json"
        },

        {
            local:
                TRAINING_STATE_FILE,

            remote:
                "GELERNT/training-state.json"
        }
    ];

    const results = [];

    for (
        const file of files
    ) {
        if (
            !fileExists(
                file.local
            )
        ) {
            continue;
        }

        try {
            await uploadFileToGitHub(
                file.local,
                file.remote,
                "LUMORA: " +
                    file.remote +
                    " aktualisieren"
            );

            results.push({
                file:
                    file.remote,

                success:
                    true
            });
        } catch (error) {
            results.push({
                file:
                    file.remote,

                success:
                    false,

                error:
                    error.message
            });
        }
    }

    const failed =
        results.filter(
            item =>
                !item.success
        );

    return {
        enabled: true,

        success:
            failed.length === 0,

        results
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

        trainingState:
            fileExists(
                TRAINING_STATE_FILE
            )
    };
}

function getTrainingState() {
    const saved =
        safeReadJSON(
            TRAINING_STATE_FILE
        );

    if (saved) {
        return {
            ...trainingState,
            ...saved,
            running:
                trainingState.running
        };
    }

    return trainingState;
}


/* =========================================================
   TRAINING
   ========================================================= */

function stopTrainingWorker() {
    if (
        trainingWorker
    ) {
        try {
            trainingWorker.postMessage(
                {
                    type:
                        "stop"
                }
            );
        } catch {}
    }
}

async function finishTraining(
    message
) {
    const modelExists =
        fileExists(
            MODEL_FILE
        );

    const tokenizerExists =
        fileExists(
            TOKENIZER_FILE
        );

    if (
        !modelExists ||
        !tokenizerExists
    ) {
        trainingState.running =
            false;

        trainingState.phase =
            "incomplete";

        trainingState.error =
            "Training beendet, aber Dateien fehlen: " +
            [
                !modelExists
                    ? "GELERNT/model.json"
                    : null,

                !tokenizerExists
                    ? "GELERNT/tokenizer.json"
                    : null
            ]
                .filter(Boolean)
                .join(", ");

        trainingState.message =
            trainingState.error;

        return;
    }

    trainingState.running =
        false;

    trainingState.phase =
        "finished";

    trainingState.finishedAt =
        new Date().toISOString();

    trainingState.error =
        null;

    trainingState.message =
        message ||
        "Training abgeschlossen.";

    try {
        const result =
            await syncLearnedToGitHub();

        trainingState.github =
            result;

        if (
            result.success === false
        ) {
            trainingState.message +=
                " GitHub-Synchronisierung hatte Fehler.";
        }
    } catch (error) {
        trainingState.github = {
            enabled:
                true,

            success:
                false,

            error:
                error.message
        };
    }

    try {
        fs.writeFileSync(
            TRAINING_STATE_FILE,
            JSON.stringify(
                trainingState,
                null,
                2
            ),
            "utf8"
        );
    } catch {}
}

function startTraining(
    payload
) {
    if (
        trainingWorker
    ) {
        return {
            success:
                false,

            error:
                "Training läuft bereits."
        };
    }

    if (
        !fileExists(
            TRAIN_WORKER
        )
    ) {
        return {
            success:
                false,

            error:
                "train-worker.js fehlt."
        };
    }

    trainingState = {
        running: true,
        phase: "starting",
        progress: 0,
        epoch: 0,
        step: 0,
        loss: null,
        message:
            "Training wird gestartet...",
        startedAt:
            new Date().toISOString(),
        finishedAt: null,
        error: null
    };

    const options =
        payload &&
        typeof payload === "object"
            ? payload
            : {};

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

                        options:
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
                    trainingState.phase =
                        message.phase ||
                        "starting";

                    trainingState.message =
                        "Training gestartet.";
                }

                else if (
                    message.type ===
                    "progress"
                ) {
                    trainingState.phase =
                        message.phase ||
                        trainingState.phase;

                    trainingState.epoch =
                        Number(
                            message.epoch ||
                            trainingState.epoch ||
                            0
                        );

                    trainingState.step =
                        Number(
                            message.step ||
                            trainingState.step ||
                            0
                        );

                    if (
                        message.loss !==
                        undefined
                    ) {
                        trainingState.loss =
                            message.loss;
                    }

                    if (
                        message.totalEpochs
                    ) {
                        trainingState.progress =
                            Math.min(
                                99,
                                Math.round(
                                    (
                                        trainingState.epoch /
                                        message.totalEpochs
                                    ) *
                                        100
                                )
                            );
                    }

                    trainingState.message =
                        message.message ||
                        (
                            "Training: " +
                            trainingState.phase
                        );
                }

                else if (
                    message.type ===
                    "epoch"
                ) {
                    trainingState.epoch =
                        message.epoch ||
                        trainingState.epoch;

                    trainingState.step =
                        message.step ||
                        trainingState.step;

                    trainingState.loss =
                        message.loss ??
                        trainingState.loss;

                    trainingState.message =
                        "Epoche " +
                        trainingState.epoch +
                        " abgeschlossen.";
                }

                else if (
                    message.type ===
                    "finished" ||
                    message.type ===
                    "complete" ||
                    message.type ===
                    "done"
                ) {
                    trainingState.epoch =
                        message.epoch ||
                        trainingState.epoch;

                    trainingState.step =
                        message.step ||
                        trainingState.step;

                    trainingState.loss =
                        message.loss ??
                        trainingState.loss;

                    await finishTraining(
                        message.message
                    );

                    trainingWorker =
                        null;
                }

                else if (
                    message.type ===
                    "stopped"
                ) {
                    trainingState.running =
                        false;

                    trainingState.phase =
                        "stopped";

                    trainingState.message =
                        message.message ||
                        "Training gestoppt.";

                    trainingWorker =
                        null;
                }

                else if (
                    message.type ===
                    "error"
                ) {
                    trainingState.running =
                        false;

                    trainingState.phase =
                        "error";

                    trainingState.error =
                        message.error ||
                        "Unbekannter Trainingsfehler.";

                    trainingState.message =
                        trainingState.error;

                    trainingWorker =
                        null;
                }
            }
        );

        trainingWorker.on(
            "error",
            error => {
                trainingState.running =
                    false;

                trainingState.phase =
                    "error";

                trainingState.error =
                    error.message;

                trainingState.message =
                    error.message;

                trainingWorker =
                    null;
            }
        );

        trainingWorker.on(
            "exit",
            code => {
                if (
                    trainingWorker
                ) {
                    trainingWorker =
                        null;
                }

                if (
                    code !== 0 &&
                    trainingState.running
                ) {
                    trainingState.running =
                        false;

                    trainingState.phase =
                        "error";

                    trainingState.error =
                        "Trainings-Worker beendet mit Code " +
                        code;

                    trainingState.message =
                        trainingState.error;
                }
            }
        );

        return {
            success:
                true,

            message:
                "Training gestartet."
        };

    } catch (error) {
        trainingWorker =
            null;

        trainingState.running =
            false;

        trainingState.phase =
            "error";

        trainingState.error =
            error.message;

        trainingState.message =
            error.message;

        return {
            success:
                false,

            error:
                error.message
        };
    }
}


/* =========================================================
   GENERIERUNG
   ========================================================= */

function startGeneration(
    payload
) {
    if (
        generateWorker
    ) {
        return {
            success:
                false,

            error:
                "Eine Generation läuft bereits."
        };
    }

    if (
        !fileExists(
            GENERATE_WORKER
        )
    ) {
        return {
            success:
                false,

            error:
                "generate-worker.js fehlt."
        };
    }

    const modelExists =
        fileExists(
            MODEL_FILE
        );

    const tokenizerExists =
        fileExists(
            TOKENIZER_FILE
        );

    if (
        !modelExists ||
        !tokenizerExists
    ) {
        return {
            success:
                false,

            error:
                "LUMORA wurde noch nicht vollständig trainiert. Es fehlen: " +
                [
                    !modelExists
                        ? "GELERNT/model.json"
                        : null,

                    !tokenizerExists
                        ? "GELERNT/tokenizer.json"
                        : null
                ]
                    .filter(Boolean)
                    .join(", ")
        };
    }

    generationState = {
        running: true,
        text: "",
        error: null
    };

    const prompt =
        String(
            payload.prompt ||
            ""
        );

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

                        prompt:
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
                    "status"
                ) {
                    return;
                }

                if (
                    message.type ===
                    "token"
                ) {
                    if (
                        typeof message.text ===
                        "string"
                    ) {
                        generationState.text =
                            message.text;
                    }
                }

                else if (
                    message.type ===
                    "complete"
                ) {
                    generationState.running =
                        false;

                    if (
                        typeof message.answer ===
                        "string"
                    ) {
                        generationState.text =
                            message.answer;
                    }

                    if (
                        payload &&
                        typeof payload.onComplete ===
                        "function"
                    ) {
                        try {
                            payload.onComplete(
                                message
                            );
                        } catch {}
                    }

                    generateWorker =
                        null;
                }

                else if (
                    message.type ===
                    "error"
                ) {
                    generationState.running =
                        false;

                    generationState.error =
                        message.error ||
                        "Generierungsfehler.";

                    generateWorker =
                        null;
                }
            }
        );

        generateWorker.on(
            "error",
            error => {
                generationState.running =
                    false;

                generationState.error =
                    error.message;

                generateWorker =
                    null;
            }
        );

        generateWorker.on(
            "exit",
            () => {
                generateWorker =
                    null;
            }
        );

        return {
            success:
                true,

            message:
                "Generation gestartet."
        };

    } catch (error) {
        generateWorker =
            null;

        generationState.running =
            false;

        generationState.error =
            error.message;

        return {
            success:
                false,

            error:
                error.message
        };
    }
}

function stopGeneration() {
    if (
        generateWorker
    ) {
        try {
            generateWorker.postMessage(
                {
                    type:
                        "stop"
                }
            );
        } catch {}

        return {
            success:
                true,

            message:
                "Generation wird gestoppt."
        };
    }

    return {
        success:
            false,

        message:
            "Keine Generation läuft."
    };
}


/* =========================================================
   DATEIEN
   ========================================================= */

function listFilesRecursive(
    directory,
    relative = ""
) {
    if (
        !fs.existsSync(
            directory
        )
    ) {
        return [];
    }

    const result = [];

    for (
        const entry of
        fs.readdirSync(
            directory,
            {
                withFileTypes:
                    true
            }
        )
    ) {
        const full =
            path.join(
                directory,
                entry.name
            );

        const rel =
            path.join(
                relative,
                entry.name
            );

        if (
            entry.isDirectory()
        ) {
            result.push(
                ...listFilesRecursive(
                    full,
                    rel
                )
            );
        } else {
            result.push(
                rel
            );
        }
    }

    return result;
}


/* =========================================================
   HTTP
   ========================================================= */

const server =
    http.createServer(
        async (
            req,
            res
        ) => {
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

                            "Access-Control-Allow-Methods":
                                "GET,POST,OPTIONS",

                            "Access-Control-Allow-Headers":
                                "Content-Type"
                        }
                    );

                    res.end();
                    return;
                }

                const url =
                    new URL(
                        req.url,
                        `http://${req.headers.host}`
                    );

                const pathname =
                    url.pathname;

                /* -----------------------------------------
                   STATUS
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/status" &&
                    req.method ===
                        "GET"
                ) {
                    jsonResponse(
                        res,
                        200,
                        {
                            name:
                                "LUMORA",

                            server:
                                true,

                            model:
                                getModelStatus(),

                            training:
                                getTrainingState(),

                            generation:
                                generationState,

                            github: {
                                configured:
                                    Boolean(
                                        GITHUB_TOKEN
                                    ),

                                repository:
                                    `${GITHUB_OWNER}/${GITHUB_REPO}`,

                                branch:
                                    GITHUB_BRANCH
                            }
                        }
                    );

                    return;
                }

                /* -----------------------------------------
                   FILES
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/files" &&
                    req.method ===
                        "GET"
                ) {
                    jsonResponse(
                        res,
                        200,
                        {
                            daten:
                                listFilesRecursive(
                                    DATA_DIR
                                ),

                            gelernt:
                                listFilesRecursive(
                                    LEARNED_DIR
                                )
                        }
                    );

                    return;
                }

                /* -----------------------------------------
                   DATEN
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/data" &&
                    req.method ===
                        "GET"
                ) {
                    const files =
                        listFilesRecursive(
                            DATA_DIR
                        );

                    const result = [];

                    for (
                        const relative of
                        files
                    ) {
                        const file =
                            path.join(
                                DATA_DIR,
                                relative
                            );

                        try {
                            const stat =
                                fs.statSync(
                                    file
                                );

                            result.push({
                                file:
                                    relative,

                                size:
                                    stat.size,

                                modified:
                                    stat.mtime
                            });
                        } catch {}
                    }

                    jsonResponse(
                        res,
                        200,
                        {
                            files:
                                result
                        }
                    );

                    return;
                }

                /* -----------------------------------------
                   MODEL
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/model" &&
                    req.method ===
                        "GET"
                ) {
                    if (
                        !fileExists(
                            MODEL_FILE
                        )
                    ) {
                        jsonResponse(
                            res,
                            404,
                            {
                                error:
                                    "GELERNT/model.json wurde noch nicht erstellt."
                            }
                        );

                        return;
                    }

                    jsonResponse(
                        res,
                        200,
                        safeReadJSON(
                            MODEL_FILE
                        )
                    );

                    return;
                }

                /* -----------------------------------------
                   TOKENIZER
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/tokenizer" &&
                    req.method ===
                        "GET"
                ) {
                    if (
                        !fileExists(
                            TOKENIZER_FILE
                        )
                    ) {
                        jsonResponse(
                            res,
                            404,
                            {
                                error:
                                    "GELERNT/tokenizer.json wurde noch nicht erstellt."
                            }
                        );

                        return;
                    }

                    jsonResponse(
                        res,
                        200,
                        safeReadJSON(
                            TOKENIZER_FILE
                        )
                    );

                    return;
                }

                /* -----------------------------------------
                   CONFIG
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/config" &&
                    req.method ===
                        "GET"
                ) {
                    jsonResponse(
                        res,
                        200,
                        safeReadJSON(
                            CONFIG_FILE
                        ) || {}
                    );

                    return;
                }

                /* -----------------------------------------
                   TRAINING STATE
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/training-state" &&
                    req.method ===
                        "GET"
                ) {
                    jsonResponse(
                        res,
                        200,
                        getTrainingState()
                    );

                    return;
                }

                /* -----------------------------------------
                   TRAIN START
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/train/start" &&
                    req.method ===
                        "POST"
                ) {
                    const payload =
                        await readRequestBody(
                            req
                        );

                    const result =
                        startTraining(
                            payload
                        );

                    jsonResponse(
                        res,
                        result.success
                            ? 200
                            : 409,
                        result
                    );

                    return;
                }

                /* -----------------------------------------
                   TRAIN STOP
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/train/stop" &&
                    req.method ===
                        "POST"
                ) {
                    stopTrainingWorker();

                    jsonResponse(
                        res,
                        200,
                        {
                            success:
                                true,

                            message:
                                "Stop-Signal an Training gesendet."
                        }
                    );

                    return;
                }

                /* -----------------------------------------
                   GENERATE
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/generate" &&
                    req.method ===
                        "POST"
                ) {
                    const payload =
                        await readRequestBody(
                            req
                        );

                    const result =
                        startGeneration(
                            payload
                        );

                    jsonResponse(
                        res,
                        result.success
                            ? 200
                            : 409,
                        result
                    );

                    return;
                }

                /* -----------------------------------------
                   GENERATE STATUS
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/generate/status" &&
                    req.method ===
                        "GET"
                ) {
                    jsonResponse(
                        res,
                        200,
                        generationState
                    );

                    return;
                }

                /* -----------------------------------------
                   GENERATE STOP
                   ----------------------------------------- */

                if (
                    pathname ===
                        "/api/generate/stop" &&
                    req.method ===
                        "POST"
                ) {
                    jsonResponse(
                        res,
                        200,
                        stopGeneration()
                    );

                    return;
                }

                /* -----------------------------------------
                   INDEX
                   ----------------------------------------- */

                if (
                    pathname === "/" ||
                    pathname ===
                        "/index.html"
                ) {
                    const index =
                        path.join(
                            ROOT,
                            "index.html"
                        );

                    if (
                        !fileExists(
                            index
                        )
                    ) {
                        textResponse(
                            res,
                            404,
                            "index.html fehlt."
                        );

                        return;
                    }

                    const html =
                        fs.readFileSync(
                            index,
                            "utf8"
                        );

                    textResponse(
                        res,
                        200,
                        html,
                        "text/html; charset=utf-8"
                    );

                    return;
                }

                /* -----------------------------------------
                   NICHT GEFUNDEN
                   ----------------------------------------- */

                jsonResponse(
                    res,
                    404,
                    {
                        error:
                            "Route nicht gefunden."
                    }
                );

            } catch (error) {
                console.error(
                    "SERVER ERROR:",
                    error
                );

                jsonResponse(
                    res,
                    500,
                    {
                        error:
                            error.message
                    }
                );
            }
        }
    );


/* =========================================================
   START
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
            "GitHub:     " +
                (
                    GITHUB_TOKEN
                        ? "konfiguriert"
                        : "nicht konfiguriert"
                )
        );

        console.log(
            "======================================"
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
            "POST /api/train/start"
        );

        console.log(
            "POST /api/train/stop"
        );

        console.log(
            "POST /api/generate"
        );

        console.log(
            "GET  /api/generate/status"
        );

        console.log(
            "POST /api/generate/stop"
        );

        console.log(
            "======================================"
        );
    }
);

"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const https = require("https");
const { Worker } = require("worker_threads");

const ROOT = __dirname;

const DATA_DIR =
    path.join(
        ROOT,
        "DATEN"
    );

const LEARNED_DIR =
    path.join(
        ROOT,
        "GELERNT"
    );

const MODEL_FILE =
    path.join(
        LEARNED_DIR,
        "model.json"
    );

const TOKENIZER_FILE =
    path.join(
        LEARNED_DIR,
        "tokenizer.json"
    );

const CONFIG_FILE =
    path.join(
        LEARNED_DIR,
        "config.json"
    );

const STATE_FILE =
    path.join(
        LEARNED_DIR,
        "training-state.json"
    );

const PORT =
    Number(
        process.env.PORT ||
        10000
    );

const GITHUB_TOKEN =
    process.env.GITHUB_TOKEN ||
    "";

const GITHUB_OWNER =
    process.env.GITHUB_OWNER ||
    "08152";

const GITHUB_REPO =
    process.env.GITHUB_REPO ||
    "cubego";

const GITHUB_BRANCH =
    process.env.GITHUB_BRANCH ||
    "main";

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


/* =========================================================
   VERZEICHNISSE
========================================================= */

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


/* =========================================================
   GLOBALER STATUS
========================================================= */

let trainingWorker = null;

let generateWorker = null;

let startupReady = false;

let startupRestorePromise = null;


let trainingState = {
    running:
        false,

    phase:
        "idle",

    progress:
        0,

    epoch:
        0,

    step:
        0,

    loss:
        null,

    totalEpochs:
        0,

    message:
        "Bereit."
};


let generationState = {
    running:
        false,

    text:
        "",

    error:
        null,

    startedAt:
        null,

    finishedAt:
        null
};


/* =========================================================
   DATEI-HILFEN
========================================================= */

function fileExists(
    file
) {
    try {
        return fs.existsSync(
            file
        );
    } catch {
        return false;
    }
}


function safeFileSize(
    file
) {
    try {
        return fs.statSync(
            file
        ).size;
    } catch {
        return 0;
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


function writeJSON(
    file,
    data
) {
    fs.mkdirSync(
        path.dirname(
            file
        ),
        {
            recursive:
                true
        }
    );

    const temp =
        file +
        ".tmp";

    fs.writeFileSync(
        temp,
        JSON.stringify(
            data,
            null,
            2
        ),
        "utf8"
    );

    fs.renameSync(
        temp,
        file
    );
}


/* =========================================================
   HTTP BODY
========================================================= */

function readBody(
    req
) {

    return new Promise(
        (
            resolve,
            reject
        ) => {

            let body =
                "";

            let settled =
                false;


            function fail(
                error
            ) {

                if (settled) {
                    return;
                }

                settled =
                    true;

                reject(
                    error
                );
            }


            req.on(
                "data",
                chunk => {

                    body += chunk;


                    if (
                        body.length >
                        10 *
                        1024 *
                        1024
                    ) {

                        fail(
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

                    if (settled) {
                        return;
                    }

                    settled =
                        true;


                    if (
                        !body.trim()
                    ) {

                        resolve(
                            {}
                        );

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
                fail
            );
        }
    );
}


/* =========================================================
   HTTP RESPONSE
========================================================= */

function json(
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


    res.end(
        body
    );
}


function text(
    res,
    status,
    value
) {

    res.writeHead(
        status,
        {
            "Content-Type":
                "text/plain; charset=utf-8",

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


    res.end(
        String(
            value
        )
    );
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
            success:
                false,

            error:
                message,

            details:
                error
                    ? error.message
                    : undefined
        }
    );
}


/* =========================================================
   DATEN
========================================================= */

function getDataFiles() {

    if (
        !fileExists(
            DATA_DIR
        )
    ) {
        return [];
    }


    const files =
        [];


    function walk(
        directory,
        relative = ""
    ) {

        let entries =
            [];


        try {

            entries =
                fs.readdirSync(
                    directory,
                    {
                        withFileTypes:
                            true
                    }
                );

        } catch {

            return;
        }


        for (
            const entry of
            entries
        ) {

            const fullPath =
                path.join(
                    directory,
                    entry.name
                );


            const relativePath =
                path.join(
                    relative,
                    entry.name
                );


            if (
                entry.isDirectory()
            ) {

                walk(
                    fullPath,
                    relativePath
                );

                continue;
            }


            if (
                entry.isFile() &&
                entry.name
                    .toLowerCase()
                    .endsWith(".json")
            ) {

                files.push(
                    relativePath
                );
            }
        }
    }


    walk(
        DATA_DIR
    );


    return files.sort(
        (
            a,
            b
        ) =>
            a.localeCompare(
                b,
                "de",
                {
                    numeric:
                        true,

                    sensitivity:
                        "base"
                }
            )
    );
}


/* =========================================================
   GITHUB
========================================================= */

function githubJSONRequest(
    method,
    apiPath,
    body = null
) {

    return new Promise(
        (
            resolve,
            reject
        ) => {

            if (
                !GITHUB_TOKEN
            ) {

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


            const request =
                https.request(
                    {
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
                                "2022-11-28",

                            ...(payload
                                ? {
                                      "Content-Type":
                                          "application/json",

                                      "Content-Length":
                                          Buffer.byteLength(
                                              payload
                                          )
                                  }
                                : {})
                        },

                        timeout:
                            120000
                    },

                    response => {

                        const chunks =
                            [];


                        response.on(
                            "data",
                            chunk => {

                                chunks.push(
                                    Buffer.from(
                                        chunk
                                    )
                                );
                            }
                        );


                        response.on(
                            "end",
                            () => {

                                const buffer =
                                    Buffer.concat(
                                        chunks
                                    );


                                const raw =
                                    buffer.toString(
                                        "utf8"
                                    );


                                let parsed =
                                    {};


                                try {

                                    parsed =
                                        raw
                                            ? JSON.parse(
                                                  raw
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

                                    return;
                                }


                                reject(
                                    new Error(
                                        "GitHub API " +
                                        response.statusCode +
                                        ": " +
                                        (
                                            parsed.message ||
                                            raw ||
                                            "Unbekannter Fehler"
                                        )
                                    )
                                );
                            }
                        );
                    }
                );


            request.on(
                "timeout",
                () => {

                    request.destroy(
                        new Error(
                            "GitHub-Anfrage Timeout."
                        )
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


function githubPath(
    filePath
) {

    return filePath
        .split("/")
        .map(
            encodeURIComponent
        )
        .join("/");
}


/* =========================================================
   GITHUB DATEI-METADATEN
========================================================= */

async function getGitHubFileMeta(
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
        githubPath(
            filePath
        ) +
        "?ref=" +
        encodeURIComponent(
            GITHUB_BRANCH
        );


    return githubJSONRequest(
        "GET",
        api
    );
}


/* =========================================================
   GITHUB GROSSE DATEI DOWNLOAD
========================================================= */

/*
   WICHTIG:

   Wir verwenden NICHT mehr:

       metadata.content

   von /contents.

   Stattdessen:
   1. SHA der Datei holen
   2. Git Blob laden
   3. Base64 decodieren

   Dadurch funktioniert auch eine größere model.json.
*/

async function downloadGitHubFile(
    filePath
) {

    const meta =
        await getGitHubFileMeta(
            filePath
        );


    if (
        !meta ||
        !meta.sha
    ) {

        throw new Error(
            "GitHub-Datei besitzt keine SHA: " +
            filePath
        );
    }


    const api =
        "/repos/" +
        encodeURIComponent(
            GITHUB_OWNER
        ) +
        "/" +
        encodeURIComponent(
            GITHUB_REPO
        ) +
        "/git/blobs/" +
        encodeURIComponent(
            meta.sha
        );


    const blob =
        await githubJSONRequest(
            "GET",
            api
        );


    if (
        !blob ||
        blob.encoding !==
            "base64" ||
        typeof blob.content !==
            "string"
    ) {

        throw new Error(
            "GitHub-Blob enthält keinen Base64-Inhalt: " +
            filePath
        );
    }


    const cleanBase64 =
        blob.content.replace(
            /\n/g,
            ""
        );


    const buffer =
        Buffer.from(
            cleanBase64,
            "base64"
        );


    if (
        !buffer.length
    ) {

        throw new Error(
            "GitHub-Datei ist leer: " +
            filePath
        );
    }


    return {
        buffer,
        sha:
            meta.sha,
        size:
            buffer.length
    };
}


/* =========================================================
   GITHUB UPLOAD
========================================================= */

async function uploadGitHubFile(
    filePath,
    buffer,
    message
) {

    if (
        !Buffer.isBuffer(
            buffer
        )
    ) {

        buffer =
            Buffer.from(
                buffer
            );
    }


    if (
        !buffer.length
    ) {

        throw new Error(
            "Leere Datei kann nicht hochgeladen werden: " +
            filePath
        );
    }


    let sha =
        null;


    try {

        const existing =
            await getGitHubFileMeta(
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
        githubPath(
            filePath
        );


    const body = {
        message:
            message ||
            "LUMORA: Datei aktualisiert",

        content:
            buffer.toString(
                "base64"
            ),

        branch:
            GITHUB_BRANCH
    };


    if (sha) {
        body.sha =
            sha;
    }


    return githubJSONRequest(
        "PUT",
        api,
        body
    );
}


/* =========================================================
   GITHUB WIEDERHERSTELLUNG
========================================================= */

async function restoreLearnedFromGitHub() {

    console.log(
        "------------------------------------"
    );

    console.log(
        "[GITHUB] Prüfe gespeicherte Dateien..."
    );


    if (
        !GITHUB_TOKEN
    ) {

        console.log(
            "[GITHUB] Kein GITHUB_TOKEN vorhanden."
        );


        return {
            success:
                false,

            restored:
                [],

            missing:
                [],

            failed:
                []
        };
    }


    const files = [
        "model.json",
        "tokenizer.json",
        "config.json",
        "training-state.json"
    ];


    const restored =
        [];

    const missing =
        [];

    const failed =
        [];


    for (
        const filename
        of files
    ) {

        const remotePath =
            "GELERNT/" +
            filename;


        const localPath =
            path.join(
                LEARNED_DIR,
                filename
            );


        try {

            const result =
                await downloadGitHubFile(
                    remotePath
                );


            fs.writeFileSync(
                localPath,
                result.buffer
            );


            restored.push(
                filename
            );


            console.log(
                "[GITHUB] Wiederhergestellt: " +
                filename +
                " (" +
                result.size +
                " Bytes)"
            );


        } catch (error) {

            if (
                String(
                    error.message
                ).includes(
                    "GitHub API 404"
                )
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


    const result = {
        success:
            failed.length === 0,

        restored,

        missing,

        failed
    };


    console.log(
        "[GITHUB] Wiederherstellung abgeschlossen:",
        result
    );


    return result;
}


/* =========================================================
   GELERNTE DATEIEN ZU GITHUB
========================================================= */

async function syncLearnedToGitHub() {

    if (
        !GITHUB_TOKEN
    ) {

        return {
            success:
                false,

            message:
                "GITHUB_TOKEN fehlt.",

            uploaded:
                [],

            failed:
                []
        };
    }


    const files = [
        "model.json",
        "tokenizer.json",
        "config.json",
        "training-state.json"
    ];


    const uploaded =
        [];

    const failed =
        [];


    for (
        const filename
        of files
    ) {

        const local =
            path.join(
                LEARNED_DIR,
                filename
            );


        if (
            !fileExists(
                local
            )
        ) {

            continue;
        }


        try {

            const buffer =
                fs.readFileSync(
                    local
                );


            if (
                !buffer.length
            ) {

                throw new Error(
                    "Lokale Datei ist leer: " +
                    filename
                );
            }


            await uploadGitHubFile(
                "GELERNT/" +
                    filename,

                buffer,

                "LUMORA: " +
                    filename +
                    " aktualisiert"
            );


            uploaded.push({
                filename,
                bytes:
                    buffer.length
            });


            console.log(
                "[GITHUB] Hochgeladen: " +
                filename +
                " (" +
                buffer.length +
                " Bytes)"
            );


        } catch (error) {

            failed.push({
                file:
                    filename,

                error:
                    error.message
            });


            console.error(
                "[GITHUB] Upload-Fehler " +
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
   MODEL STATUS
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

        modelBytes:
            safeFileSize(
                MODEL_FILE
            ),

        tokenizerBytes:
            safeFileSize(
                TOKENIZER_FILE
            ),

        modelFile:
            MODEL_FILE,

        tokenizerFile:
            TOKENIZER_FILE
    };
}


/* =========================================================
   GESAMTSTATUS
========================================================= */

function getStatus() {

    return {
        success:
            true,

        name:
            "LUMORA",

        online:
            true,

        ready:
            startupReady,

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
                getDataFiles()
                    .length
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
   TRAINING STARTEN
========================================================= */

function startTraining(
    payload = {}
) {

    if (
        trainingWorker
    ) {

        return {
            success:
                false,

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
            success:
                false,

            message:
                "train-worker.js wurde nicht gefunden."
        };
    }


    const options = {

        epochs:
            Math.max(
                1,
                Number(
                    payload.epochs ||
                    20
                )
            ),

        sequenceLength:
            Math.max(
                16,
                Number(
                    payload.sequenceLength ||
                    256
                )
            ),

        learningRate:
            Number(
                payload.learningRate ||
                0.00025
            ),

        contextSize:
            Math.max(
                16,
                Number(
                    payload.contextSize ||
                    256
                )
            ),

        embeddingSize:
            Math.max(
                32,
                Number(
                    payload.embeddingSize ||
                    128
                )
            ),

        layers:
            Math.max(
                1,
                Number(
                    payload.layers ||
                    4
                )
            ),

        heads:
            Math.max(
                1,
                Number(
                    payload.heads ||
                    4
                )
            ),

        headSize:
            Math.max(
                8,
                Number(
                    payload.headSize ||
                    32
                )
            ),

        feedForwardSize:
            Math.max(
                64,
                Number(
                    payload.feedForwardSize ||
                    512
                )
            )
    };


    trainingState = {

        running:
            true,

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

        startedAt:
            new Date()
                .toISOString(),

        options
    };


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


        return {
            success:
                false,

            message:
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
                    "started" ||

                message.type ===
                    "progress" ||

                message.type ===
                    "epoch"
            ) {

                trainingState = {

                    ...trainingState,

                    ...message,

                    running:
                        true
                };


                if (
                    message.type ===
                    "epoch"
                ) {

                    trainingState.phase =
                        "training";
                }


                return;
            }


            if (
                message.type ===
                "warning"
            ) {

                trainingState.warning =
                    message.message;

                return;
            }


            if (
                message.type ===
                "finished"
            ) {

                const worker =
                    trainingWorker;


                trainingState = {

                    ...trainingState,

                    ...message,

                    running:
                        false,

                    phase:
                        "finished",

                    progress:
                        100,

                    message:
                        message.message ||
                        "Training abgeschlossen.",

                    finishedAt:
                        new Date()
                            .toISOString()
                };


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

                const worker =
                    trainingWorker;


                trainingState = {

                    ...trainingState,

                    ...message,

                    running:
                        false,

                    phase:
                        "stopped",

                    message:
                        message.message ||
                        "Training gestoppt.",

                    finishedAt:
                        new Date()
                            .toISOString()
                };


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

                const worker =
                    trainingWorker;


                trainingState = {

                    ...trainingState,

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
                        message.stack,

                    finishedAt:
                        new Date()
                            .toISOString()
                };


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
                    error.message,

                finishedAt:
                    new Date()
                        .toISOString()
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
                        code,

                    error:
                        "Worker exit " +
                        code,

                    finishedAt:
                        new Date()
                            .toISOString()
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


/* =========================================================
   TRAINING STOP
========================================================= */

async function stopTraining() {

    if (
        !trainingWorker
    ) {

        return {

            success:
                false,

            message:
                "Kein Training läuft."
        };
    }


    trainingState.message =
        "Training wird gestoppt...";


    try {

        trainingWorker.postMessage(
            {
                type:
                    "stop"
            }
        );

    } catch {}


    return {

        success:
            true,

        stopping:
            true
    };
}


/* =========================================================
   GENERATION START
========================================================= */

function startGeneration(
    payload = {}
) {

    if (
        generateWorker
    ) {

        return {

            success:
                false,

            message:
                "Generation läuft bereits."
        };
    }


    if (
        !startupReady
    ) {

        return {

            success:
                false,

            message:
                "LUMORA wird noch aus GitHub wiederhergestellt. Bitte danach erneut versuchen."
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

            message:
                "generate-worker.js wurde nicht gefunden."
        };
    }


    if (
        !fileExists(
            MODEL_FILE
        ) ||
        !fileExists(
            TOKENIZER_FILE
        )
    ) {

        return {

            success:
                false,

            message:
                "Modell oder Tokenizer fehlt. Trainiere LUMORA zuerst."
        };
    }


    const prompt =
        String(
            payload.prompt ||
            ""
        ).trim();


    if (!prompt) {

        return {

            success:
                false,

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
            new Date()
                .toISOString(),

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

        generationState = {

            ...generationState,

            running:
                false,

            error:
                error.message,

            finishedAt:
                new Date()
                    .toISOString()
        };


        generateWorker =
            null;


        return {

            success:
                false,

            message:
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

                /*
                  Wenn generate-worker bereits
                  den kompletten aktuellen Text schickt,
                  direkt übernehmen.

                  Dadurch landen keine numerischen
                  Token-IDs im Status-Text.
                */

                if (
                    typeof message.text ===
                    "string"
                ) {

                    generationState.text =
                        message.text;

                } else if (
                    typeof message.token ===
                    "string"
                ) {

                    generationState.text +=
                        message.token;
                }


                return;
            }


            if (
                message.type ===
                "status"
            ) {

                generationState.message =
                    message.message;

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
                        typeof message.text ===
                        "string"

                            ? message.text

                            : generationState.text,

                    answer:
                        message.answer,

                    tokens:
                        message.tokens,

                    promptTokens:
                        message.promptTokens,

                    finishedAt:
                        new Date()
                            .toISOString()
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
                        message.error ||
                        "Unbekannter Generierungsfehler.",

                    finishedAt:
                        new Date()
                            .toISOString()
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
                    new Date()
                        .toISOString()
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
                        new Date()
                            .toISOString()
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


/* =========================================================
   GENERATION STOP
========================================================= */

function stopGeneration() {

    if (
        !generateWorker
    ) {

        return {

            success:
                false,

            message:
                "Keine Generation läuft."
        };
    }


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

        stopping:
            true
    };
}


/* =========================================================
   HTTP SERVER
========================================================= */

const server =
    http.createServer(
        async (
            req,
            res
        ) => {

            try {

                /* -------------------------
                   CORS
                ------------------------- */

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


                /* =====================================================
                   STATUS
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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


                /* =====================================================
                   DATEIEN
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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


                /* =====================================================
                   DATENINFO
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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


                        try {

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

                        } catch {}
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


                /* =====================================================
                   MODEL
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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

                            bytes:
                                safeFileSize(
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


                /* =====================================================
                   TOKENIZER
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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

                            bytes:
                                safeFileSize(
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


                /* =====================================================
                   CONFIG
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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


                /* =====================================================
                   TRAINING STATUS
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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


                /* =====================================================
                   GENERATION STATUS
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
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


                /* =====================================================
                   MANUELLES RELOAD
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
                    pathname ===
                        "/api/reload"
                ) {

                    try {

                        startupRestorePromise =
                            restoreLearnedFromGitHub();


                        const github =
                            await startupRestorePromise;


                        startupReady =
                            true;


                        json(
                            res,
                            200,
                            {

                                success:
                                    true,

                                github,

                                status:
                                    getStatus()
                            }
                        );

                    } catch (error) {

                        startupReady =
                            true;


                        json(
                            res,
                            500,
                            {

                                success:
                                    false,

                                error:
                                    error.message,

                                status:
                                    getStatus()
                            }
                        );
                    }


                    return;
                }


                /* =====================================================
                   TRAIN START
                ===================================================== */

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


                /* =====================================================
                   TRAIN STOP
                ===================================================== */

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


                /* =====================================================
                   GENERATE
                ===================================================== */

                if (
                    req.method ===
                        "POST" &&
                    pathname ===
                        "/api/generate"
                ) {

                    if (
                        startupRestorePromise
                    ) {

                        try {

                            await startupRestorePromise;

                        } catch {}
                    }


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


                /* =====================================================
                   GENERATE STOP
                ===================================================== */

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


                /* =====================================================
                   HEALTH
                ===================================================== */

                if (
                    req.method ===
                        "GET" &&
                    pathname ===
                        "/health"
                ) {

                    json(
                        res,
                        200,
                        {

                            ok:
                                true,

                            ready:
                                startupReady
                        }
                    );


                    return;
                }


                /* =====================================================
                   STATIC FILES
                ===================================================== */

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

                    let decoded;


                    try {

                        decoded =
                            decodeURIComponent(
                                pathname
                            );

                    } catch {

                        text(
                            res,
                            400,
                            "Ungültiger Pfad."
                        );

                        return;
                    }


                    const safePath =
                        path.normalize(
                            decoded
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


                const rootPrefix =
                    rootResolved +
                    path.sep;


                if (
                    fileResolved !==
                        rootResolved &&
                    !fileResolved.startsWith(
                        rootPrefix
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
                        fileResolved
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
                        fileResolved
                    ).toLowerCase();


                const mimeTypes = {

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
                };


                const mime =
                    mimeTypes[
                        extension
                    ] ||
                    "application/octet-stream";


                res.writeHead(
                    200,
                    {

                        "Content-Type":
                            mime,

                        "Cache-Control":
                            "no-cache",

                        "Access-Control-Allow-Origin":
                            "*"
                    }
                );


                fs.createReadStream(
                    fileResolved
                ).pipe(
                    res
                );


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
            "GET  /api/reload"
        );

        console.log(
            "GET  /health"
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


        /*
          GitHub-Wiederherstellung läuft
          nach dem Öffnen des Ports.

          Dadurch meldet Render seinen Port
          sofort als aktiv.
        */

        startupRestorePromise =
            restoreLearnedFromGitHub()
                .catch(
                    error => {

                        console.error(
                            "[GITHUB] Start-Wiederherstellung fehlgeschlagen:",
                            error.message
                        );


                        return {
                            success:
                                false,

                            restored:
                                [],

                            missing:
                                [],

                            failed: [
                                {
                                    file:
                                        "startup",

                                    error:
                                        error.message
                                }
                            ]
                        };
                    }
                )
                .finally(
                    () => {

                        startupReady =
                            true;


                        console.log(
                            "[START] LUMORA bereit."
                        );


                        console.log(
                            "[START] model.json: " +
                            safeFileSize(
                                MODEL_FILE
                            ) +
                            " Bytes"
                        );
                    }
                );
    }
);


/* =========================================================
   SHUTDOWN
========================================================= */

function shutdown(
    signal
) {

    console.log(
        signal +
        " erhalten."
    );


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
    ).unref();
}


process.on(
    "SIGTERM",
    () =>
        shutdown(
            "SIGTERM"
        )
);


process.on(
    "SIGINT",
    () =>
        shutdown(
            "SIGINT"
        )
);

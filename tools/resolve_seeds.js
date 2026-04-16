#!/usr/bin/env node
/**
 * Resolve TF2 market inspect URLs to paint_kit_seed values via the TF2 Game
 * Coordinator.
 *
 * Protocol (from TF2 leaked source, tf_gcmessages.proto lines 197-198 and 2065-2080):
 *   Request:  k_EMsgGC_Client2GCEconPreviewDataBlockRequest = 6402
 *             CMsgGC_Client2GCEconPreviewDataBlockRequest {
 *               optional uint64 param_s = 1;  // Steam owner ID   (S-type URL)
 *               optional uint64 param_a = 2;  // Asset ID
 *               optional uint64 param_d = 3;  // D param          (ownership proof)
 *               optional uint64 param_m = 4;  // Market listing  (M-type URL)
 *             }
 *   Response: k_EMsgGC_Client2GCEconPreviewDataBlockResponse = 6403
 *             -> iteminfo.econitem.original_id (field 16) = paint_kit_seed
 *
 * Rate limit: the TF2 client enforces ~2.5 s between requests.
 *
 * Usage:
 *   node resolve_seeds.js <config.json>
 *
 * config.json:
 *   {
 *     "username":         "your_steam_login",
 *     "password":         "your_password",
 *     "two_factor_code":  "optional mobile Steam Guard code",
 *     "input":            "inspect_urls.csv",
 *     "output":           "seeds.csv",
 *     "request_delay_ms": 2500
 *   }
 *
 * inspect_urls.csv has a `listing_id,inspect_url` header; output is
 * `listing_id,seed` CSV (missing listings have empty seed field).
 */

"use strict";

const SteamUser = require("steam-user");
const fs        = require("fs");
const path      = require("path");

// Force line-buffered output -- on Windows, Node may block-buffer stdout
// when it's piped through a parent process, which hides live progress.
if (process.stdout._handle && process.stdout._handle.setBlocking) {
    process.stdout._handle.setBlocking(true);
}
if (process.stderr._handle && process.stderr._handle.setBlocking) {
    process.stderr._handle.setBlocking(true);
}

function log(msg) {
    const t = new Date().toISOString().slice(11, 19);
    process.stdout.write(`[${t}] ${msg}\n`);
}
function logErr(msg) {
    const t = new Date().toISOString().slice(11, 19);
    process.stderr.write(`[${t}] ${msg}\n`);
}

// -----------------------------------------------------------------------------
// Minimal protobuf encode/decode (the two messages we touch are tiny).
// -----------------------------------------------------------------------------

// Encode a BigInt or Number as a protobuf varint and push bytes to `out`.
function writeVarint(out, value) {
    let v = typeof value === "bigint" ? value : BigInt(value);
    if (v < 0n) v += 1n << 64n;              // treat negatives as uint64
    while (v >= 0x80n) {
        out.push(Number((v & 0x7fn) | 0x80n));
        v >>= 7n;
    }
    out.push(Number(v));
}

// Read a varint starting at `buf[off]`. Returns {value: BigInt, off: next}.
function readVarint(buf, off) {
    let result = 0n;
    let shift  = 0n;
    for (let i = 0; i < 10; i++) {
        if (off >= buf.length) throw new Error("truncated varint");
        const b = buf[off++];
        result |= BigInt(b & 0x7f) << shift;
        if ((b & 0x80) === 0) return { value: result, off };
        shift += 7n;
    }
    throw new Error("varint too long");
}

// Read a wire tag: returns {fieldNumber, wireType, off}.
function readTag(buf, off) {
    const { value, off: o } = readVarint(buf, off);
    const tag   = Number(value);
    return { fieldNumber: tag >>> 3, wireType: tag & 0x7, off: o };
}

// Encode the inspect request (all four fields as uint64 varints, tags 1..4).
function encodePreviewRequest({ paramS, paramA, paramD, paramM }) {
    const nonzero = (v) => v != null && BigInt(v) !== 0n;
    const out = [];
    if (nonzero(paramS)) { writeVarint(out, (1 << 3) | 0); writeVarint(out, paramS); }
    if (nonzero(paramA)) { writeVarint(out, (2 << 3) | 0); writeVarint(out, paramA); }
    if (nonzero(paramD)) { writeVarint(out, (3 << 3) | 0); writeVarint(out, paramD); }
    if (nonzero(paramM)) { writeVarint(out, (4 << 3) | 0); writeVarint(out, paramM); }
    return Buffer.from(out);
}

// Walk a protobuf message and return the VALUE of the first occurrence of a
// field (by number) at the given path of (fieldNumber, wireType) steps.
// `path` is [submsgFieldNum, submsgFieldNum, ..., targetFieldNum].
// All intermediate steps must be wireType 2 (length-delimited submessages).
function extractVarintAtPath(buf, path) {
    let cur = buf;
    for (let step = 0; step < path.length; step++) {
        const field = path[step];
        const lastStep = step === path.length - 1;
        let off = 0;
        let found = false;
        while (off < cur.length) {
            const tag = readTag(cur, off);
            off = tag.off;
            if (tag.fieldNumber === field) {
                if (lastStep) {
                    // Expect varint (wireType 0) for uint64 target
                    if (tag.wireType !== 0) {
                        throw new Error(
                            `field ${field} wire type ${tag.wireType}, expected varint`);
                    }
                    const { value } = readVarint(cur, off);
                    return value;
                } else {
                    // Expect length-delimited submessage
                    if (tag.wireType !== 2) {
                        throw new Error(
                            `field ${field} wire type ${tag.wireType}, expected submsg`);
                    }
                    const { value: len, off: o2 } = readVarint(cur, off);
                    const L = Number(len);
                    cur = cur.slice(o2, o2 + L);
                    off = 0;
                    found = true;
                    break;
                }
            } else {
                // Skip unrelated field
                off = skipField(cur, off, tag.wireType);
            }
        }
        if (!found && !lastStep) return null;
        if (off >= cur.length && !found && lastStep) return null;
    }
    return null;
}

function skipField(buf, off, wireType) {
    switch (wireType) {
        case 0: { // varint
            const { off: o } = readVarint(buf, off);
            return o;
        }
        case 1: return off + 8;    // fixed64
        case 2: {                  // length-delimited
            const { value, off: o } = readVarint(buf, off);
            return o + Number(value);
        }
        case 5: return off + 4;    // fixed32
        default:
            throw new Error(`unknown wire type ${wireType}`);
    }
}

// -----------------------------------------------------------------------------
// Inspect URL parsing
// -----------------------------------------------------------------------------

function parseInspectUrl(url) {
    // TF2 inspect URL patterns (after url-decoding spaces):
    //   S<steamid>A<assetid>D<dispatch>   (inventory item)
    //   M<listingid>A<assetid>D<dispatch> (market listing)
    const decoded = decodeURIComponent(url);
    let m = decoded.match(/S(\d+)A(\d+)D(\d+)/);
    if (m) {
        return { paramS: m[1], paramA: m[2], paramD: m[3], paramM: "0" };
    }
    m = decoded.match(/M(\d+)A(\d+)D(\d+)/);
    if (m) {
        return { paramS: "0", paramA: m[2], paramD: m[3], paramM: m[1] };
    }
    return null;
}

// -----------------------------------------------------------------------------
// CSV I/O
// -----------------------------------------------------------------------------

function readCsv(filepath) {
    const text = fs.readFileSync(filepath, "utf8");
    const lines = text.split(/\r?\n/).filter(Boolean);
    if (!lines.length) return [];
    const header = lines[0].split(",").map(s => s.trim());
    return lines.slice(1).map(line => {
        const fields = line.split(",");
        const row = {};
        header.forEach((h, i) => { row[h] = (fields[i] || "").trim(); });
        return row;
    });
}

function writeCsv(filepath, rows, columns) {
    const lines = [columns.join(",")];
    for (const r of rows) {
        lines.push(columns.map(c => r[c] ?? "").join(","));
    }
    fs.writeFileSync(filepath, lines.join("\n") + "\n");
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

async function main() {
    log("resolver starting");
    const configPath = process.argv[2];
    if (!configPath) {
        logErr("usage: node resolve_seeds.js <config.json>");
        process.exit(2);
    }
    log(`reading config: ${configPath}`);
    const config = JSON.parse(fs.readFileSync(configPath, "utf8"));

    if (!config.input || !config.output) {
        logErr("config requires 'input' and 'output' fields");
        process.exit(2);
    }
    const inputRows = readCsv(path.resolve(config.input));
    log(`loaded ${inputRows.length} listings from ${config.input}`);
    if (!inputRows.length) {
        writeCsv(config.output, [], ["listing_id", "seed"]);
        process.exit(0);
    }

    const delay_ms = Number(config.request_delay_ms ?? 2500);

    // Persist session between runs so Steam Guard isn't needed every time.
    // Default to ~/.bloodscope-steam but allow override via config.
    const defaultDataDir = path.join(
        process.env.HOME || process.env.USERPROFILE || ".", ".bloodscope-steam");
    const dataDir = config.data_dir || defaultDataDir;
    try { fs.mkdirSync(dataDir, { recursive: true }); } catch (_) {}
    log(`steam-user dataDirectory: ${dataDir}`);

    const client = new SteamUser({ dataDirectory: dataDir });

    const loginDetails = {
        accountName: config.username,
        password:    config.password,
    };
    if (config.two_factor_code) {
        loginDetails.twoFactorCode = config.two_factor_code;
    }

    let tf2Ready = false;
    const gcQueue = [];

    // Watchdog -- if the whole pipeline stalls for a minute, print why.
    let lastProgress = Date.now();
    const watchdog = setInterval(() => {
        const idle = Math.floor((Date.now() - lastProgress) / 1000);
        if (idle >= 30) {
            log(`(still running, ${idle}s since last event; ` +
                `tf2Ready=${tf2Ready}, gcQueue=${gcQueue.length})`);
        }
    }, 15000);

    client.on("error", (err) => {
        logErr(`steam-user error: ${err.message}`);
        clearInterval(watchdog);
        process.exit(1);
    });

    client.on("disconnected", (eresult, msg) => {
        logErr(`steam-user disconnected: eresult=${eresult} ${msg || ""}`);
    });

    client.on("steamGuard", (domain, cb, lastCodeWrong) => {
        const where = domain ? `email (${domain})` : "mobile app";
        if (lastCodeWrong) logErr("previous Steam Guard code was rejected");
        logErr(`Steam Guard required via ${where}.`);
        logErr("  Put a fresh code in config.two_factor_code (or --steam-2fa) and rerun.");
        clearInterval(watchdog);
        process.exit(1);
    });

    client.on("loggedOn", () => {
        lastProgress = Date.now();
        log(`Steam login OK (${client.steamID.getSteamID64()})`);
        log("setting game to TF2 (appid 440)");
        client.setPersona(SteamUser.EPersonaState.Online);
        client.gamesPlayed([440]);
    });

    client.on("appLaunched", (app) => {
        if (app === 440) {
            lastProgress = Date.now();
            log("TF2 launched, waiting for GC welcome");
        }
    });

    // TF2 GC welcome message
    client.on("receivedFromGC", (appid, msgType, payload) => {
        if (appid !== 440) return;
        lastProgress = Date.now();
        if (msgType === 4004) {          // k_EMsgGCClientWelcome
            if (!tf2Ready) {
                tf2Ready = true;
                log(`TF2 GC welcome received (payload ${payload.length} bytes), ` +
                    "starting inspect loop");
                runInspectLoop().catch(err => {
                    logErr(`fatal: ${err.stack || err.message}`);
                    clearInterval(watchdog);
                    process.exit(1);
                });
            }
        } else if (msgType === 6403) {   // Preview data block response
            const pending = gcQueue.shift();
            if (pending) pending.resolve(payload);
            else log(`WARN: GC 6403 received with empty queue (${payload.length} bytes)`);
        }
    });

    async function inspectOne(req) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(
                () => reject(new Error("GC inspect timeout (15s no reply)")),
                15000);
            gcQueue.push({
                resolve: (buf) => { clearTimeout(timer); resolve(buf); },
                reject:  (err) => { clearTimeout(timer); reject(err); },
            });
            const body = encodePreviewRequest(req);
            client.sendToGC(440, 6402, {}, body);
        });
    }

    async function runInspectLoop() {
        const results = [];
        log(`starting ${inputRows.length} inspect requests (${delay_ms} ms between)`);
        const t0 = Date.now();
        for (let i = 0; i < inputRows.length; i++) {
            const row = inputRows[i];
            const lid = row.listing_id || row.listingid || "";
            const url = row.inspect_url || row.inspect || "";
            const parsed = parseInspectUrl(url);
            const tag = `[${i + 1}/${inputRows.length}]`;
            if (!parsed) {
                log(`${tag} ${lid}  BAD URL, skipping`);
                results.push({ listing_id: lid, seed: "" });
                continue;
            }
            const paramsStr = `S=${parsed.paramS} A=${parsed.paramA} ` +
                              `D=${parsed.paramD} M=${parsed.paramM}`;
            log(`${tag} ${lid}  sending inspect (${paramsStr})`);
            try {
                const t = Date.now();
                const resp = await inspectOne(parsed);
                lastProgress = Date.now();
                const ms   = Date.now() - t;
                const seed = extractVarintAtPath(resp, [1, 1, 16]);
                if (seed == null) {
                    // Dump first 64 hex bytes of the response for debugging
                    const hex = resp.slice(0, 64).toString("hex");
                    log(`${tag} ${lid}  reply in ${ms}ms but NO original_id; ` +
                        `resp[0..64]=${hex}...`);
                    results.push({ listing_id: lid, seed: "" });
                } else {
                    log(`${tag} ${lid}  seed=${seed} (${ms}ms)`);
                    results.push({ listing_id: lid, seed: seed.toString() });
                }
            } catch (err) {
                log(`${tag} ${lid}  ERROR: ${err.message}`);
                results.push({ listing_id: lid, seed: "" });
            }
            if (i + 1 < inputRows.length) {
                await new Promise(r => setTimeout(r, delay_ms));
            }
        }
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        writeCsv(path.resolve(config.output), results, ["listing_id", "seed"]);
        const ok = results.filter(r => r.seed).length;
        log(`DONE. ${ok}/${results.length} resolved in ${elapsed}s`);
        log(`wrote ${config.output}`);
        clearInterval(watchdog);
        client.logOff();
        setTimeout(() => process.exit(0), 1000);
    }

    log(`logging in as ${config.username}${config.two_factor_code ? " with 2FA" : ""}`);
    client.logOn(loginDetails);
}

main().catch(err => {
    console.error(err.stack || err.message);
    process.exit(1);
});

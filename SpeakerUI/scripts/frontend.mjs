import http from 'node:http';
import https from 'node:https';
import { createReadStream } from 'node:fs';
import { stat, realpath } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { pipeline } from 'node:stream';
const root = await realpath(resolve('dist'));
const host = process.env.FRONTEND_HOST || '127.0.0.1';
const port = Number(process.env.FRONTEND_PORT || 3000);
const backend = new URL(process.env.BACKEND_URL || 'http://127.0.0.1:3001');
const request = backend.protocol === 'https:' ? https.request : http.request;
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.json': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2' };
function proxy(req, res) {
    const route = new URL(req.url, 'http://localhost'), target = new URL(backend);
    target.pathname = route.pathname;
    target.search = route.search;
    const upstream = request(target, { method: req.method, headers: req.headers }, reply => { res.writeHead(reply.statusCode ?? 502, reply.headers); pipeline(reply, res, () => { }); });
    upstream.on('error', () => { if (!res.headersSent)
        res.writeHead(503, { 'Content-Type': 'application/json' }); res.end('{"error":"backend_unavailable"}'); });
    req.on('aborted', () => upstream.destroy());
    res.on('close', () => upstream.destroy());
    req.pipe(upstream);
}
const server = http.createServer(async (req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (path.startsWith('/api/') || path.startsWith('/auth/')) {
        proxy(req, res);
        return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.writeHead(405);
        res.end();
        return;
    }
    try {
        const decoded = decodeURIComponent(path), candidate = resolve(root, '.' + (decoded === '/' ? '/index.html' : decoded));
        if (!candidate.startsWith(root + sep)) {
            res.writeHead(403);
            res.end();
            return;
        }
        let file = await realpath(candidate);
        if (!file.startsWith(root + sep)) {
            res.writeHead(403);
            res.end();
            return;
        }
        const type = types[extname(file)] || 'application/octet-stream';
        let encoding;
        for (const item of ['br', 'gzip']) {
            if ((req.headers['accept-encoding'] || '').split(',').some(v => v.trim().split(';')[0] === item && !/;\s*q=0(?:\.0*)?(?:\s|;|$)/.test(v))) {
                const compressed = file + (item === 'br' ? '.br' : '.gz');
                try {
                    await stat(compressed);
                    file = compressed;
                    encoding = item;
                    break;
                }
                catch (error) {
                    if (error.code !== 'ENOENT')
                        throw error;
                }
            }
        }
        const info = await stat(file);
        if (!info.isFile()) {
            res.writeHead(404);
            res.end();
            return;
        }
        res.writeHead(200, { 'Content-Type': type, 'Content-Length': info.size, 'Cache-Control': path.startsWith('/assets/') ? 'public, max-age=31536000, immutable' : 'no-cache', 'X-Content-Type-Options': 'nosniff', Vary: 'Accept-Encoding', ...(encoding ? { 'Content-Encoding': encoding } : {}) });
        if (req.method === 'HEAD')
            res.end();
        else
            pipeline(createReadStream(file), res, () => { });
    }
    catch (error) {
        res.writeHead(error.code === 'ENOENT' ? 404 : 400);
        res.end('Not found');
    }
});
server.on('upgrade', (req, socket, head) => {
    if (req.url !== '/ws') {
        socket.destroy();
        return;
    }
    const upstream = request(new URL('/ws', backend), { headers: req.headers });
    upstream.on('upgrade', (reply, remote, remoteHead) => { socket.write(`HTTP/1.1 ${reply.statusCode} ${reply.statusMessage}\r\n` + Object.entries(reply.headers).map(([k, v]) => `${k}: ${v}\r\n`).join('') + '\r\n'); if (head.length)
        remote.write(head); if (remoteHead.length)
        socket.write(remoteHead); socket.pipe(remote); remote.pipe(socket); socket.on('error', () => remote.destroy()); remote.on('error', () => socket.destroy()); socket.on('close', () => remote.destroy()); remote.on('close', () => socket.destroy()); });
    upstream.on('response', () => socket.destroy());
    upstream.on('error', () => socket.destroy());
    upstream.end();
});
server.listen(port, host, () => console.log(`Ambient frontend: http://${host}:${port}; backend=${backend.origin}`));
for (const signal of ['SIGINT', 'SIGTERM'])
    process.once(signal, () => server.close(() => process.exit(0)));

import Fastify from 'fastify';
import { loadDummyRuntime, parseFixtureRequest } from './inference/dummy-runtime.js';
import websocket from '@fastify/websocket';
import staticFiles from '@fastify/static';
import { readFile } from 'node:fs/promises';
import { Readable } from 'node:stream';
import type { ReadableStream as NodeReadableStream } from 'node:stream/web';
import { resolve } from 'node:path';
import type { Config } from './config.js';
import { AUDIO, Intent, MediaSource, type Command, type Track } from '../shared/protocol.js';
import { createLocalMedia, parseRange } from './media/local.js';
import { createHomeAssistant } from './integrations/home-assistant.js';
import { createSpotify } from './integrations/spotify.js';
import { IntegrationError } from './integrations/http.js';
import { lightCommand, spotifyCommand } from './integrations/commands.js';
import { searchMedia } from './media/search.js';
import { attachVoiceSession, type ActionResult, type RecordingLease } from './inference/session.js';
import type { InferenceFactory } from './inference/adapter.js';
import { PythonRuntime } from './inference/python-runtime.js';
import { RuntimeActions } from './inference/runtime-actions.js';

interface Options { staticDir?: string; fetcher?: typeof fetch; inferenceFactory?: InferenceFactory }

export async function createApp(config: Config, options: Options = {}) {
  const https = config.tls ? { cert: await readFile(config.tls.cert), key: await readFile(config.tls.key) } : undefined;
  const app = Fastify({ ...(https ? { https } : {}), bodyLimit: 4_096, requestTimeout: 10_000, logger: false });
  const dummyRuntime = loadDummyRuntime();
  const runtime = config.runtime ? new PythonRuntime(config.runtime) : undefined;
  const local = await createLocalMedia(config.mediaDir);
  const fetcher = options.fetcher ?? fetch;
  const ha = createHomeAssistant(config.ha, fetcher);
  const spotify = createSpotify(config.spotify, config.dataDir, fetcher);
  const actions=new RuntimeActions({enabled:config.actionsEnabled,bindings:config.bindings},{
    light:c=>ha.control(c),spotify:c=>spotify.control(c),song:uri=>spotify.playConfigured(uri),
    volume:async()=>{const state=await spotify.snapshot();return state.playback?.device?.volume??undefined;},
  });
  let actionChain=Promise.resolve();
  runtime?.on('event',event=>{
    if(event.event!=='command')return;
    actionChain=actionChain.then(async()=>{
      const execution=await actions.handle(event);
      if(execution)runtime.emit('action',{event:'execution',utterance_id:event.utterance_id,execution});
    }).catch(()=>{});
  });
  const lease: RecordingLease = { owner: undefined };
  const activeSockets = new Set<import('ws').default>();
  const cameraStreams = new Map<AbortController, { entityId: string; session?: string }>();

  app.addHook('onRequest', async (req, reply) => {
    reply.header('X-Content-Type-Options', 'nosniff').header('Referrer-Policy', 'no-referrer');
    if (req.headers.host !== new URL(config.origin).host && req.headers.host !== `${config.host}:${config.port}`) return reply.code(403).send({ error: 'Unexpected Host.' });
    const origin = req.headers.origin;
    if (origin && origin !== config.origin) return reply.code(403).send({ error: 'Origin is not allowed.' });
    if ((req.method !== 'GET' && req.method !== 'HEAD') && origin !== config.origin) return reply.code(403).send({ error: 'A same-origin request is required.' });
    if (req.url.startsWith('/api') || req.url.startsWith('/auth')) reply.header('Cache-Control', 'no-store');
  });
  app.setErrorHandler((error, _req, reply) => {
    if (error instanceof IntegrationError) {
      const status = error.code === 'disabled' ? 503 : error.code === 'not_found' ? 404 : error.code === 'invalid_range' ? 416 : error.code.startsWith('invalid') ? 400 : 502;
      return reply.code(status).send({ error: error.code, message: error.message, outcomeUnknown: error.outcomeUnknown });
    }
    if (error instanceof Error && 'validation' in error) return reply.code(400).send({ error: 'invalid_request' });
    return reply.code(500).send({ error: 'internal_error', message: 'Request failed. Check configuration and local file access.' });
  });

  async function playback(source: MediaSource, id: string): Promise<ActionResult> {
    if (source === MediaSource.Local) {
      if (!local.find(id)) throw new IntegrationError('not_found', 'Unknown local track.');
      return { status: 'succeeded', detail: 'Play this local track in the browser.', data: { url: `/api/media/local/${id}` } };
    }
    const track = spotify.find(id);
    if (!track) throw new IntegrationError('invalid_target', 'Search for and select the Spotify track first.');
    try { await spotify.play(id); return { status: 'succeeded', detail: 'Spotify accepted playback on the selected device.' }; }
    catch (error) {
      if (!(error instanceof IntegrationError)) throw error;
      return { status: error.outcomeUnknown ? 'unknown' : 'failed', detail: `${error.message} Select a local match explicitly to use it instead.`, data: { tracks: local.search(track.title) } };
    }
  }
  async function dispatch(command: Command): Promise<ActionResult> {
    try {
      switch (command.intent) {
        case Intent.LightOn: case Intent.LightOff:
          await ha.setLight(command.entityId, command.intent === Intent.LightOn);
          return { status: 'succeeded', detail: 'Home Assistant accepted the light action; observed state updates separately.' };
        case Intent.CameraSnapshot:
          return { status: 'succeeded', detail: 'Request the selected camera snapshot.', data: { url: `/api/home-assistant/camera/${encodeURIComponent(command.entityId)}` } };
        case Intent.MediaSearch:
          return { status: 'succeeded', detail: 'Media search completed.', data: await searchMedia(command.query, local, spotify) };
        case Intent.MediaPlay: return await playback(command.source, command.id);
        case Intent.MediaPause:
          if (command.source === MediaSource.Spotify) await spotify.pause();
          return { status: 'succeeded', detail: 'Playback pause requested.', data: { pauseLocal: command.source === MediaSource.Local } };
      }
    } catch (error) {
      if (!(error instanceof IntegrationError)) throw error;
      return { status: error.outcomeUnknown ? 'unknown' : 'failed', detail: error.message };
    }
  }
  await app.register(websocket, { options: { maxPayload: 4_096, perMessageDeflate: false } });
  app.get('/ws', { websocket: true }, (socket, req) => {
    if (runtime) { socket.close(1008, 'Use the host microphone or WAV upload in Inference debug.'); return; }
    if (req.headers.origin !== config.origin || activeSockets.size >= 4) { socket.close(1008, 'Origin rejected or connection limit reached.'); return; }
    activeSockets.add(socket); socket.once('close', () => activeSockets.delete(socket));
    attachVoiceSession(socket, config, lease, dispatch, options.inferenceFactory);
  });
  app.get('/api/health', async () => ({ status: 'ok', inference: config.inferenceMode, instance: config.runtime?.token.slice(0,12), runtime: runtime?.status() ?? dummyRuntime.status(), actionsEnabled: config.actionsEnabled, audio: AUDIO, homeAssistant: ha.snapshot().status, spotify: config.spotify.enabled ? 'configured' : 'disabled', localMedia: local.status() }));
  if (runtime) {
    app.get('/ws/runtime', { websocket: true }, (socket, req) => {
      if (req.headers.origin !== config.origin || activeSockets.size >= 4) { socket.close(1008, 'Origin rejected or connection limit reached'); return; }
      activeSockets.add(socket);
      const send = (event: object) => { if (socket.readyState === 1) { if (socket.bufferedAmount > 256_000) socket.close(1013, 'Slow consumer'); else socket.send(JSON.stringify(event)); } };
      runtime.on('event', send);runtime.on('action',send); const current = runtime.current(); if (current) send(current);
      socket.once('close', () => { runtime.off('event', send);runtime.off('action',send); activeSockets.delete(socket); });
    });
    app.get('/api/runtime/devices', async (_req,reply) => {
      try{return await runtime.request('devices');}
      catch(error){return reply.code(503).send({error:'microphone_discovery',message:error instanceof Error?error.message.slice(0,500):'Microphone discovery failed.'});}
    });
    app.post<{ Body: { device: number } }>('/api/runtime/listen', {
      schema: { body: { type:'object',required:['device'],additionalProperties:false,properties:{device:{type:'integer',minimum:0}} } },
    }, async (req,reply) => {
      try{return await runtime.request('listen', { device: req.body.device });}
      catch(error){return reply.code(503).send({error:'microphone_capture',message:error instanceof Error?error.message.slice(0,500):'Microphone capture failed.'});}
    });
    app.post('/api/runtime/mute', async () => runtime.request('mute'));
    app.addContentTypeParser('audio/wav', { parseAs: 'buffer', bodyLimit: 4*1024*1024 }, (_req, body, done) => done(null, body));
    app.post<{ Querystring: { wake?: string } }>('/api/runtime/wav', { bodyLimit: 4*1024*1024 }, async (req, reply) => {
      if (!Buffer.isBuffer(req.body)) return reply.code(400).send({ error: 'WAV body required' });
      try { return await runtime.request('wav', { data: req.body.toString('base64'), wake: req.query.wake === 'true' }); }
      catch (error) { return reply.code(422).send({ error: error instanceof Error ? error.message : 'WAV inference failed' }); }
    });
    app.post('/api/runtime/shutdown', async (req, reply) => {
      if (req.headers['x-launch-token'] !== config.runtime?.token) return reply.code(403).send({ error: 'Unknown installation' });
      setImmediate(() => { void app.close(); }); return { stopping: true };
    });
  }
  app.post('/api/ambient/infer', async (req, reply) => {
    if (runtime) return reply.code(409).send({ error: 'Simulator disabled in real inference mode' });
    const fixture = parseFixtureRequest(req.body);
    if (fixture === undefined) return reply.code(400).send({ error: 'invalid_fixture' });
    return dummyRuntime.infer(fixture);
  });
  app.get('/api/home-assistant/entities', async () => ({ ...ha.snapshot(), actionsEnabled: config.actionsEnabled }));
  app.get('/api/spotify/state', async () => ({ ...await spotify.snapshot(), actionsEnabled: config.actionsEnabled }));
  app.post('/api/home-assistant/control', async (req, reply) => {
    if (!config.actionsEnabled) return reply.code(409).send({ error: 'actions_disabled', message: 'Device actions are disabled.' });
    await ha.control(lightCommand(req.body));
    return { ...ha.snapshot(), actionsEnabled: config.actionsEnabled };
  });
  app.post('/api/spotify/control', async (req, reply) => {
    if (!config.actionsEnabled) return reply.code(409).send({ error: 'actions_disabled', message: 'Playback actions are disabled.' });
    await spotify.control(spotifyCommand(req.body));
    return { ...await spotify.snapshot(), actionsEnabled: config.actionsEnabled };
  });
  app.get<{ Params: { id: string } }>('/api/home-assistant/camera/:id', async (req, reply) => {
    const camera = await ha.camera(req.params.id);
    return reply.type(camera.contentType).send(Readable.fromWeb(camera.body as NodeReadableStream<Uint8Array>));
  });
  app.get<{ Params: { id: string }; Querystring: { session?: string } }>('/api/home-assistant/camera/:id/stream', {
    schema: { querystring: { type: 'object', properties: { session: { type: 'string', pattern: '^[a-zA-Z0-9-]{1,64}$' } }, additionalProperties: false } },
  }, async (req, reply) => {
    if (cameraStreams.size >= 2) return reply.code(429).send({ error: 'camera_busy', message: 'Close another camera view and try again.' });
    const lifetime = new AbortController();
    cameraStreams.set(lifetime, { entityId: req.params.id, session: req.query.session });
    const close = () => {
      lifetime.abort(); cameraStreams.delete(lifetime);
      req.raw.off('aborted', close); reply.raw.off('close', close);
    };
    req.raw.once('aborted', close); reply.raw.once('close', close);
    try {
      const camera = await ha.cameraStream(req.params.id, lifetime.signal);
      const body = Readable.fromWeb(camera.body as NodeReadableStream<Uint8Array>, { signal: lifetime.signal });
      return reply.type(camera.contentType).send(body);
    } catch (error) { close(); throw error; }
  });
  // Chromium can retain the last MJPEG frame after EOF without an image error.
  // A per-view status read lets that view clear its frame when its own stream ends.
  app.get<{ Params: { id: string; session: string } }>('/api/home-assistant/camera/:id/stream-status/:session', async (req) => ({
    active: [...cameraStreams.values()].some(stream => stream.entityId === req.params.id && stream.session === req.params.session),
  }));
  app.get<{ Querystring: { q: string } }>('/api/media/search', {
    schema: { querystring: { type: 'object', required: ['q'], properties: { q: { type: 'string', minLength: 1, maxLength: 200 } }, additionalProperties: false } },
  }, async (req) => searchMedia(req.query.q.trim(), local, spotify));
  app.post<{ Body: Pick<Track, 'id' | 'source'> }>('/api/media/play', {
    schema: { body: { type: 'object', required: ['id', 'source'], properties: { id: { type: 'string', minLength: 1, maxLength: 64 }, source: { enum: ['local', 'spotify'] } }, additionalProperties: false } },
  }, async (req, reply) => {
    if (req.body.source === MediaSource.Spotify && !config.actionsEnabled) return reply.code(409).send({ error: 'actions_disabled' });
    return playback(req.body.source, req.body.id);
  });
  app.post('/api/media/pause', async (_req, reply) => {
    if (!config.actionsEnabled) return reply.code(409).send({ error: 'actions_disabled' });
    return dispatch({ intent: Intent.MediaPause, source: MediaSource.Spotify });
  });
  app.get<{ Params: { id: string } }>('/api/media/local/:id', async (req, reply) => {
    let opened;
    try { opened = await local.open(req.params.id); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') throw new IntegrationError('not_found', 'Track no longer exists.');
      throw error;
    }
    const { handle, size } = opened;
    reply.header('Accept-Ranges', 'bytes').type('audio/mpeg');
    try {
      const range = parseRange(req.headers.range, size);
      if (range) reply.code(206).header('Content-Range', `bytes ${range.start}-${range.end}/${size}`).header('Content-Length', range.end - range.start + 1);
      else reply.header('Content-Length', size);
      if (req.method === 'HEAD' || size === 0) { await handle.close(); return reply.send(); }
      return reply.send(handle.createReadStream(range ? { start: range.start, end: range.end, autoClose: true } : { autoClose: true }));
    } catch (error) { await handle.close(); if (error instanceof IntegrationError && error.code === 'invalid_range') reply.header('Content-Range', `bytes */${size}`); throw error; }
  });
  app.post('/auth/spotify', async (_req, reply) => {
    const auth = spotify.authorize();
    reply.header('Set-Cookie', `spotify_oauth=${auth.binding}; HttpOnly; SameSite=Lax; Path=/auth/spotify; Max-Age=600${config.origin.startsWith('https:') ? '; Secure' : ''}`);
    return { url: auth.url };
  });
  app.get<{ Querystring: { state?: string; code?: string } }>('/auth/spotify/callback', async (req, reply) => {
    reply.header('Set-Cookie', 'spotify_oauth=; HttpOnly; SameSite=Lax; Path=/auth/spotify; Max-Age=0');
    const cookie = req.headers.cookie?.split(';').map((item) => item.trim()).find((item) => item.startsWith('spotify_oauth='))?.slice('spotify_oauth='.length);
    await spotify.callback(req.query.state || '', cookie, req.query.code);
    return reply.redirect('/');
  });
  if (options.staticDir) {
    await app.register(staticFiles, { root: resolve(options.staticDir), index: ['index.html'], cacheControl: true, maxAge: 0 });
  }
  app.addHook('onReady', async () => { ha.start(); });
  app.addHook('preClose', async () => { for (const socket of activeSockets) socket.terminate(); for (const stream of cameraStreams.keys()) stream.abort(); });
  app.addHook('onClose', async () => { ha.close(); dummyRuntime.close(); await runtime?.close(); });
  return app;
}

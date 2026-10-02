import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { createApp } from './app.js';
import { readConfig } from './config.js';

const config = readConfig(process.env, process.cwd());
const staticDir = resolve('dist');
const app = await createApp(config, process.env.SERVE_UI !== 'false' && existsSync(staticDir) ? { staticDir } : {});
await app.listen({ host: config.host, port: config.port });
console.log(`Gateway listening on ${config.tls ? 'https' : 'http'}://${config.host}:${config.port}; inference=${config.inferenceMode}; allowed browser origin=${config.origin}`);
let stopping = false;
async function shutdown() {
  if (stopping) return;
  stopping = true;
  await app.close();
}
process.once('SIGINT', () => { void shutdown(); });
process.once('SIGTERM', () => { void shutdown(); });

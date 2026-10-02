import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';

type RuntimeEvent = Record<string, any>;

/** One native CPU worker owns models and the host microphone. No action dispatch. */
export class PythonRuntime extends EventEmitter {
  private child: ChildProcessWithoutNullStreams;
  private pending = new Map<string, { resolve: (value: RuntimeEvent) => void; reject: (error: Error) => void; timer: NodeJS.Timeout }>();
  private closed = false;
  private health: RuntimeEvent = { engine: 'onnx-cpu', loaded: false, state: 'starting' };
  private snapshot: RuntimeEvent | undefined;
  constructor(options: { python: string; root: string; bundle: string; threads: number }) {
    super();
    this.child = spawn(options.python, ['-u', '-m', 'scripts.runtime.speaker_worker', '--bundle', options.bundle, '--diagnostic', '--threads', String(options.threads)],
      { cwd: options.root, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
    const lines = createInterface({ input: this.child.stdout });
    lines.on('line', line => {
      if (line.length > 1_000_000) { this.fail('Runtime event too large'); return; }
      try {
        const event = JSON.parse(line) as RuntimeEvent;
        if (event.event === 'reply') {
          const request = this.pending.get(event.id);
          if (request) { clearTimeout(request.timer); this.pending.delete(event.id); event.ok ? request.resolve(event) : request.reject(new Error(event.message)); }
          return;
        }
        if (event.event === 'ready') { this.health = event; this.snapshot = event; }
        if (event.event === 'state') { this.health = { ...this.health, state: event.state }; this.snapshot = { ...this.health, ...event, event: 'ready' }; }
        this.emit('event', event);
      } catch { this.fail('Invalid JSON from inference worker'); }
    });
    this.child.stderr.on('data', chunk => process.stderr.write(chunk));
    this.child.stdin.on('error', error => { if (!this.closed) this.fail(error.message); });
    this.child.on('error', error => this.fail(error.message));
    this.child.on('exit', code => { if (!this.closed) this.fail(`Inference worker exited (${code})`); });
  }
  status() { return { ...this.health }; }
  current() { return this.snapshot; }
  async request(op: string, payload: RuntimeEvent = {}) {
    if (this.closed || !this.health.loaded) throw new Error('Inference worker is not ready');
    if (this.pending.size >= 4) throw new Error('Runtime busy');
    const id = randomUUID();
    return new Promise<RuntimeEvent>((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('Runtime request timed out')); }, 60000);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ ...payload, id, op }) + '\n', error => { if (error) this.fail(error.message); });
    });
  }
  private fail(message: string) {
    this.health = { ...this.health, loaded: false, state: 'runtime_error' };
    for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(new Error(message)); }
    this.pending.clear(); this.emit('event', { event: 'error', message, actions_enabled: false });
  }
  async close() {
    if (this.closed) return;
    this.closed = true; this.fail('Runtime stopped');
    if (this.child.exitCode !== null || !this.child.pid) return;
    await new Promise<void>(resolve => {
      const timer = setTimeout(() => this.child.kill(), 3000);
      this.child.once('exit', () => { clearTimeout(timer); resolve(); });
      this.child.stdin.end(JSON.stringify({ op: 'shutdown' }) + '\n');
    });
  }
}

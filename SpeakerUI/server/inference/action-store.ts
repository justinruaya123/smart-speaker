import { EventEmitter } from 'node:events';
import { mkdir, open, readFile, rename, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { isRecord } from '../../shared/protocol.js';
import type { CanonicalCommand } from '../../shared/canonical-command.js';

export interface Execution { status: string; detail: string; [key: string]: any }
export interface Job {
  id: string; kind: 'timer' | 'alarm' | 'reminder'; label: string; deadline: number | null;
  status: 'scheduled' | 'active' | 'due' | 'missed' | 'paused' | 'completed' | 'cancelled';
  createdAt: number; remaining?: number; timezone?: string;
}
interface State { version: 1; revision: number; jobs: Job[]; executions: Record<string, Execution> }

/** Single server-owned journal. Claim is durable before any external side effect. */
export class ActionStore extends EventEmitter {
  private state: State = { version: 1, revision: 0, jobs: [], executions: {} };
  private chain: Promise<unknown> = Promise.resolve();
  private constructor(private directory: string | undefined, private now: () => number) { super(); }
  static memory(now = Date.now) { return new ActionStore(undefined, now); }
  static async open(directory: string, now = Date.now) {
    const store = new ActionStore(directory, now);
    try {
      const value: unknown = JSON.parse(await readFile(join(directory, 'action-state.json'), 'utf8'));
      if (!isRecord(value) || value.version !== 1 || !Number.isSafeInteger(value.revision) || !Array.isArray(value.jobs) || !isRecord(value.executions) ||
        value.jobs.some(job => !isRecord(job) || typeof job.id !== 'string' || !['timer', 'alarm', 'reminder'].includes(String(job.kind)) || typeof job.label !== 'string' ||
          !['scheduled', 'active', 'due', 'missed', 'paused', 'completed', 'cancelled'].includes(String(job.status)) ||
          !(job.deadline === null || typeof job.deadline === 'number' && Number.isFinite(job.deadline)) || typeof job.createdAt !== 'number') ||
        Object.values(value.executions).some(receipt => !isRecord(receipt) || typeof receipt.status !== 'string' || typeof receipt.detail !== 'string')) {
        throw new Error('Invalid action store; preserve action-state.json and repair it before starting.');
      }
      store.state = value as unknown as State;
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
    let changed = false;
    for (const receipt of Object.values(store.state.executions)) {
      if (receipt.status === 'pending' || receipt.status === 'requested') {
        receipt.status = 'unknown'; receipt.detail = 'Server restarted before the provider outcome was confirmed. This action will not be retried.'; changed = true;
      }
    }
    for (const job of store.state.jobs) {
      if ((job.status === 'scheduled' && job.deadline! <= now()) || job.status === 'due') { job.status = 'missed'; changed = true; }
    }
    if (changed) await store.save();
    return store;
  }
  snapshot() { return structuredClone({ revision: this.state.revision, jobs: this.state.jobs, receipts: Object.entries(this.state.executions).slice(-30).map(([id, receipt]) => ({ ...receipt, id })) }); }
  receipt(id: string) { const value = this.state.executions[id]; return value ? structuredClone(value) : undefined; }
  private async save() {
    this.state.revision++;
    if (this.directory) {
      await mkdir(this.directory, { recursive: true, mode: 0o700 });
      const temporary = join(this.directory, `.action-state-${randomUUID()}.tmp`);
      try {
        const handle = await open(temporary, 'wx', 0o600);
        try { await handle.writeFile(JSON.stringify(this.state)); await handle.sync(); } finally { await handle.close(); }
        await rename(temporary, join(this.directory, 'action-state.json'));
        if (process.platform !== 'win32') {
          const directory = await open(this.directory, 'r');
          try { await directory.sync(); } finally { await directory.close(); }
        }
      } catch (error) { await unlink(temporary).catch(() => {}); throw error; }
    }
    this.emit('change', this.snapshot());
  }
  private mutate<T>(fn: () => Promise<T>): Promise<T> {
    const pending = this.chain.then(async () => {
      const previous = structuredClone(this.state);
      try { return await fn(); } catch (error) { this.state = previous; throw error; }
    });
    this.chain = pending.catch(() => {});
    return pending;
  }
  claim(id: string, command: CanonicalCommand): Promise<boolean> {
    return this.mutate(async () => {
      if (Object.hasOwn(this.state.executions, id)) return false;
      this.state.executions[id] = { status: 'pending', detail: 'Action recorded before dispatch.', command, createdAt: this.now() };
      await this.save(); return true;
    });
  }
  finish(id: string, receipt: Execution) {
    return this.mutate(async () => { this.state.executions[id] = { ...this.state.executions[id], ...receipt, id }; await this.save(); return this.receipt(id)!; });
  }
  addJob(input: Pick<Job, 'id' | 'kind' | 'label' | 'deadline' | 'timezone'>) {
    return this.mutate(async () => {
      if (this.state.jobs.some(job => job.id === input.id)) throw new Error('Job ID already exists.');
      const job: Job = { ...input, createdAt: this.now(), status: input.deadline === null ? 'active' : 'scheduled' };
      this.state.jobs.push(job); await this.save(); return structuredClone(job);
    });
  }
  updateJob(id: string, action: string) {
    return this.mutate(async () => {
      const job = this.state.jobs.find(job => job.id === id);
      if (!job || ['completed', 'cancelled'].includes(job.status)) throw new Error('Job is no longer active.');
      if (action === 'complete') job.status = 'completed';
      else if (action === 'cancel') job.status = 'cancelled';
      else if (action === 'snooze' && ['due', 'missed'].includes(job.status)) { job.deadline = this.now() + 300000; job.status = 'scheduled'; }
      else if (action === 'pause' && job.kind === 'timer' && job.status === 'scheduled') { job.remaining = Math.max(0, job.deadline! - this.now()); job.deadline = null; job.status = 'paused'; }
      else if (action === 'resume' && job.kind === 'timer' && job.status === 'paused') { job.deadline = this.now() + job.remaining!; delete job.remaining; job.status = 'scheduled'; }
      else throw new Error('Invalid job action for the current state.');
      await this.save(); return structuredClone(job);
    });
  }
  expire(now = this.now()) {
    if (!this.state.jobs.some(job => job.status === 'scheduled' && job.deadline! <= now)) return Promise.resolve();
    return this.mutate(async () => {
      let changed = false;
      for (const job of this.state.jobs) if (job.status === 'scheduled' && job.deadline! <= now) { job.status = 'due'; changed = true; }
      if (changed) await this.save();
    });
  }
  async flush() { await this.chain; }
}

import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { opendir, open, realpath } from 'node:fs/promises';
import { basename, extname, join, relative, sep } from 'node:path';
import { MediaSource, type Track } from '../../shared/protocol.js';
import { IntegrationError } from '../integrations/http.js';

const MAX_TRACKS = 10_000;

export function parseRange(header: string | undefined, size: number): { start: number; end: number } | undefined {
  if (!header) return;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header);
  if (!match || (match[1] === '' && match[2] === '') || size === 0) throw new IntegrationError('invalid_range', 'Unsatisfiable byte range.');
  const first = match[1] ?? '';
  const last = match[2] ?? '';
  let start: number;
  let end: number;
  if (first === '') {
    const suffix = Number(last);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) throw new IntegrationError('invalid_range', 'Unsatisfiable byte range.');
    start = Math.max(0, size - suffix); end = size - 1;
  } else { start = Number(first); end = last === '' ? size - 1 : Math.min(Number(last), size - 1); }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start >= size || end < start) throw new IntegrationError('invalid_range', 'Unsatisfiable byte range.');
  return { start, end };
}

export async function createLocalMedia(directory: string) {
  const entries = new Map<string, { path: string; track: Track }>();
  let status = 'ready';
  let root: string;
  try { root = await realpath(directory); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; root = directory; status = 'folder_missing'; }
  async function scan(folder: string, depth: number): Promise<void> {
    if (depth > 8) { status = 'index_limit'; return; }
    const dir = await opendir(folder);
    for await (const entry of dir) {
      if (entries.size >= MAX_TRACKS) { status = 'index_limit'; return; }
      const path = join(folder, entry.name);
      if (entry.isDirectory()) await scan(path, depth + 1);
      if (!entry.isFile() || extname(entry.name).toLowerCase() !== '.mp3') continue;
      const id = createHash('sha256').update(relative(root, path)).digest('hex').slice(0, 32);
      entries.set(id, { path, track: { source: MediaSource.Local, id, title: basename(entry.name, extname(entry.name)) } });
    }
  }
  if (status === 'ready') await scan(root, 0);
  return {
    status() { return { status, count: entries.size }; },
    find(id: string) { return entries.get(id)?.track; },
    search(query: string): Track[] {
      const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
      return [...entries.values()].filter(({ track }) => terms.every((term) => track.title.toLocaleLowerCase().includes(term))).sort((a,b) => a.track.title.localeCompare(b.track.title)).slice(0,5).map(({ track }) => track);
    },
    async open(id: string) {
      const entry = entries.get(id);
      if (!entry) throw new IntegrationError('not_found', 'Unknown local track ID.');
      const canonical = await realpath(entry.path);
      const rel = relative(root, canonical);
      if (rel === '..' || rel.startsWith(`..${sep}`) || canonical !== entry.path) throw new IntegrationError('not_found', 'Track path has changed. Re-index the folder.');
      const handle = await open(canonical, constants.O_RDONLY | constants.O_NOFOLLOW);
      const stat = await handle.stat();
      if (!stat.isFile()) { await handle.close(); throw new IntegrationError('not_found', 'Track is not a regular file.'); }
      return { handle, size: stat.size };
    },
  };
}
export type LocalMedia = Awaited<ReturnType<typeof createLocalMedia>>;

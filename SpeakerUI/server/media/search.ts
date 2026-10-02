import type { LocalMedia } from './local.js';
import type { Spotify } from '../integrations/spotify.js';
import { IntegrationError } from '../integrations/http.js';
import type { SearchResults } from '../../shared/protocol.js';

export async function searchMedia(query: string, local: LocalMedia, spotify: Spotify): Promise<SearchResults> {
  if (!spotify.enabled) return { tracks: local.search(query), fallbackReason: 'Spotify is disabled; searching local filenames.' };
  try {
    const tracks = await spotify.search(query);
    if (tracks.length) return { tracks };
    return { tracks: local.search(query), fallbackReason: 'No Spotify matches; searching local filenames.' };
  } catch (error) {
    if (!(error instanceof IntegrationError)) throw error;
    return { tracks: local.search(query), fallbackReason: `${error.message} Searching local filenames.` };
  }
}

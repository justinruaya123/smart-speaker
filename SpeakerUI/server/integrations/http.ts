export class IntegrationError extends Error {
  constructor(readonly code: string, message: string, readonly outcomeUnknown = false) { super(message); }
}

/** Do not log provider bodies, request headers, tokens, or credential-bearing URLs. */
export async function request(fetcher: typeof fetch, url: string, init: RequestInit = {}, mutation = false): Promise<Response> {
  try {
    return await fetcher(url, { ...init, redirect: 'error', signal: AbortSignal.timeout(5_000) });
  } catch {
    throw new IntegrationError('unavailable', mutation ? 'Provider request failed; the action outcome is unknown. Do not automatically retry.' : 'Provider is unavailable or timed out.', mutation);
  }
}

export async function requireSuccess(response: Response): Promise<void> {
  if (response.ok) return;
  await response.body?.cancel().catch(() => {});
  if (response.status === 401 || response.status === 403) throw new IntegrationError('authentication', 'Provider rejected authentication or permission.');
  throw new IntegrationError('provider_error', `Provider returned HTTP ${response.status}.`);
}

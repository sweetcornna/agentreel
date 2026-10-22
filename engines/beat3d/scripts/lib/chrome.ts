// Minimal Chrome DevTools Protocol driver over Bun's built-in WebSocket. No dependencies.
// Launches the locally installed Google Chrome (CHROME_PATH, see tools.ts) headless with a throwaway
// profile directory (never the user's own profile), attaches to page targets with flattened sessions.
// GPU backend flags come from tools.ts (BEAT3D_GPU / BEAT3D_GPU_FLAGS, per-platform defaults).

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromePath, gpuFlags } from './tools';

const QUIET_FLAGS = [
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-extensions',
  '--disable-component-update',
  '--disable-background-networking',
  '--disable-sync',
  '--disable-default-apps',
  '--disable-features=Translate,OptimizationHints,MediaRouter,DialMediaRouteProvider',
  '--metrics-recording-only',
  '--mute-audio',
  '--password-store=basic',
  '--use-mock-keychain',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
  '--hide-scrollbars',
];

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void; method: string };

export class Chrome {
  private ws!: WebSocket;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Map<string, ((params: any) => void)[]>();
  private proc!: ReturnType<typeof Bun.spawn>;
  private profileDir!: string;

  static async launch(extraFlags: string[] = [], gpu: string[] = gpuFlags()): Promise<Chrome> {
    const c = new Chrome();
    c.profileDir = mkdtempSync(join(tmpdir(), 'b3d-chrome-'));
    c.proc = Bun.spawn(
      [chromePath(), ...gpu, ...QUIET_FLAGS, `--user-data-dir=${c.profileDir}`, '--remote-debugging-port=0', ...extraFlags, 'about:blank'],
      { stdout: 'ignore', stderr: 'pipe' },
    );
    const wsUrl = await readDevtoolsUrl(c.proc.stderr as ReadableStream<Uint8Array>);
    c.ws = new WebSocket(wsUrl);
    await new Promise<void>((res, rej) => {
      c.ws.onopen = () => res();
      c.ws.onerror = (e) => rej(new Error(`CDP websocket error ${String(e)}`));
    });
    c.ws.onmessage = (ev) => c.onMessage(String(ev.data));
    return c;
  }

  private onMessage(raw: string): void {
    const msg = JSON.parse(raw);
    if (msg.id !== undefined) {
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.error) p.reject(new Error(`${p.method}: ${msg.error.message}`));
      else p.resolve(msg.result);
      return;
    }
    const key = msg.sessionId ? `${msg.sessionId}:${msg.method}` : msg.method;
    for (const fn of this.listeners.get(key) ?? []) fn(msg.params);
  }

  send(method: string, params: Record<string, unknown> = {}, sessionId?: string): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }

  on(event: string, fn: (params: any) => void, sessionId?: string): void {
    const key = sessionId ? `${sessionId}:${event}` : event;
    const arr = this.listeners.get(key) ?? [];
    arr.push(fn);
    this.listeners.set(key, arr);
  }

  async newPage(width: number, height: number): Promise<Page> {
    const { targetId } = await this.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await this.send('Target.attachToTarget', { targetId, flatten: true });
    const page = new Page(this, sessionId, targetId);
    await page.send('Runtime.enable');
    await page.send('Page.enable');
    await page.send('Emulation.setDeviceMetricsOverride', { width: 640, height: 360, deviceScaleFactor: 1, mobile: false });
    return page;
  }

  async close(): Promise<void> {
    try {
      await Promise.race([this.send('Browser.close'), Bun.sleep(3000)]);
    } catch {
      /* already gone */
    }
    try {
      this.ws.close();
    } catch {
      /* ignore */
    }
    this.proc.kill();
    await this.proc.exited;
    rmSync(this.profileDir, { recursive: true, force: true });
  }
}

export class Page {
  constructor(
    readonly chrome: Chrome,
    readonly sessionId: string,
    readonly targetId: string,
  ) {}

  send(method: string, params: Record<string, unknown> = {}): Promise<any> {
    return this.chrome.send(method, params, this.sessionId);
  }

  onConsole(fn: (type: string, text: string) => void): void {
    this.chrome.on(
      'Runtime.consoleAPICalled',
      (p) => fn(p.type, (p.args ?? []).map((a: any) => a.value ?? a.description ?? '').join(' ')),
      this.sessionId,
    );
    this.chrome.on(
      'Runtime.exceptionThrown',
      (p) => fn('exception', p.exceptionDetails?.exception?.description ?? p.exceptionDetails?.text ?? 'exception'),
      this.sessionId,
    );
  }

  async goto(url: string): Promise<void> {
    const loaded = new Promise<void>((res) => this.chrome.on('Page.loadEventFired', () => res(), this.sessionId));
    await this.send('Page.navigate', { url });
    await loaded;
  }

  /** Evaluate an expression in the page; awaits promises; returns the JSON value. */
  async eval<T = unknown>(expression: string): Promise<T> {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) {
      const d = r.exceptionDetails;
      throw new Error(`page eval failed: ${d.exception?.description ?? d.text}`);
    }
    return r.result?.value as T;
  }

  async close(): Promise<void> {
    await this.chrome.send('Target.closeTarget', { targetId: this.targetId });
  }
}

async function readDevtoolsUrl(stderr: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stderr.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value);
    const m = /DevTools listening on (ws:\/\/\S+)/.exec(buf);
    if (m) {
      // Keep draining stderr so Chrome never blocks on a full pipe.
      (async () => {
        try {
          while (!(await reader.read()).done) {
            /* discard */
          }
        } catch {
          /* closed */
        }
      })();
      return m[1];
    }
  }
  throw new Error(`Chrome did not report a DevTools URL. stderr:\n${buf}`);
}

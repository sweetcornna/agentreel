// Minimal Chrome DevTools Protocol driver over Bun's built-in WebSocket. No dependencies.
// Shared ancestry with the beat3d engine's driver: launches the installed Google Chrome headless (lib/tools.ts finds
// it; CHROME_PATH overrides) with a throwaway profile directory (never the user's own profile) and attaches to page
// targets with flattened sessions.
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {chromePath} from './tools';

/**
 * GL back ends (lib/tools.ts defaultGl picks one per platform; --gl or PAPER2D_GL overrides).
 *   metal    macOS: ANGLE's Metal back end, WebGL2 on the GPU in --headless=new (measured on Apple silicon, Chrome 154–155)
 *   angle    ANGLE's default back end for the platform (--use-gl=angle; Metal on macOS, D3D11 on Windows)
 *   d3d11    Windows: ANGLE on Direct3D 11
 *   vulkan   Linux with a GPU: ANGLE on Vulkan
 *   egl      Linux with a GPU: native EGL
 *   swangle  ANGLE on SwiftShader (CPU), for machines without a GPU; not pixel-identical to the GPU back ends
 * Pictures from different back ends (or machines) are not expected to match bit for bit; determinism checks compare
 * renders made with the same back end on the same machine.
 */
export const GL_FLAGS = {
	metal: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-driver-bug-workarounds'],
	angle: ['--use-gl=angle', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl'],
	d3d11: ['--use-gl=angle', '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl'],
	vulkan: ['--use-gl=angle', '--use-angle=vulkan', '--enable-features=Vulkan', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl'],
	egl: ['--use-gl=egl', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl'],
	swangle: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--enable-webgl'],
} as const;
export type GlName = keyof typeof GL_FLAGS;

const QUIET_FLAGS = [
	'--headless=new',
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
	'--disable-ipc-flooding-protection',
	'--hide-scrollbars',
	// screenshots in sRGB whatever the display profile
	'--force-color-profile=srgb',
];

type Pending = {resolve: (v: any) => void; reject: (e: Error) => void; method: string};

export class Chrome {
	private ws!: WebSocket;
	private nextId = 1;
	private pending = new Map<number, Pending>();
	private listeners = new Map<string, ((params: any) => void)[]>();
	private proc!: ReturnType<typeof Bun.spawn>;
	private profileDir!: string;
	private closed = false;

	static async launch(gl: GlName, extraFlags: string[] = []): Promise<Chrome> {
		const c = new Chrome();
		const exe = chromePath();
		c.profileDir = mkdtempSync(join(tmpdir(), 'paper2d-chrome-'));
		c.proc = Bun.spawn(
			[exe, ...QUIET_FLAGS, ...GL_FLAGS[gl], `--user-data-dir=${c.profileDir}`, '--remote-debugging-port=0', ...extraFlags, 'about:blank'],
			{stdout: 'ignore', stderr: 'pipe'},
		);
		try {
			const wsUrl = await readDevtoolsUrl(c.proc.stderr as ReadableStream<Uint8Array>);
			c.ws = new WebSocket(wsUrl);
			await new Promise<void>((res, rej) => {
				c.ws.onopen = () => res();
				c.ws.onerror = (e) => rej(new Error(`CDP websocket error ${String(e)}`));
			});
		} catch (e) {
			c.proc.kill();
			rmSync(c.profileDir, {recursive: true, force: true});
			throw e;
		}
		c.ws.onmessage = (ev) => c.onMessage(String(ev.data));
		c.ws.onclose = () => {
			for (const p of c.pending.values()) p.reject(new Error(`${p.method}: CDP connection closed`));
			c.pending.clear();
		};
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
			this.pending.set(id, {resolve, reject, method});
			this.ws.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
		});
	}

	on(event: string, fn: (params: any) => void, sessionId?: string): () => void {
		const key = sessionId ? `${sessionId}:${event}` : event;
		const arr = this.listeners.get(key) ?? [];
		arr.push(fn);
		this.listeners.set(key, arr);
		return () => {
			const cur = this.listeners.get(key) ?? [];
			this.listeners.set(
				key,
				cur.filter((f) => f !== fn),
			);
		};
	}

	/** A new tab whose CSS viewport is width x height and whose screenshots are scaled by deviceScaleFactor. */
	async newPage(width: number, height: number, deviceScaleFactor = 1): Promise<Page> {
		// Target.createTarget does not accept width/height in headless; the viewport is set by emulation
		const {targetId} = await this.send('Target.createTarget', {url: 'about:blank'});
		const {sessionId} = await this.send('Target.attachToTarget', {targetId, flatten: true});
		const page = new Page(this, sessionId, targetId);
		await page.send('Runtime.enable');
		await page.send('Page.enable');
		await page.send('Emulation.setFocusEmulationEnabled', {enabled: true});
		await page.setViewport(width, height, deviceScaleFactor);
		return page;
	}

	async close(): Promise<void> {
		if (this.closed) return;
		this.closed = true;
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
		rmSync(this.profileDir, {recursive: true, force: true});
	}

	/** Synchronous last resort for process exit handlers. */
	killNow(): void {
		if (this.closed) return;
		this.closed = true;
		try {
			this.proc.kill();
		} catch {
			/* ignore */
		}
		try {
			rmSync(this.profileDir, {recursive: true, force: true});
		} catch {
			/* ignore */
		}
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

	setViewport(width: number, height: number, deviceScaleFactor = 1): Promise<unknown> {
		return this.send('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor, mobile: false});
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
		let off = () => {};
		const loaded = new Promise<void>((res) => {
			off = this.chrome.on('Page.loadEventFired', () => res(), this.sessionId);
		});
		const r = await this.send('Page.navigate', {url});
		if (r.errorText) throw new Error(`navigation to ${url} failed: ${r.errorText}`);
		await loaded;
		off();
	}

	/** Evaluate an expression in the page; awaits promises; returns the JSON value. */
	async eval<T = unknown>(expression: string): Promise<T> {
		const r = await this.send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
		if (r.exceptionDetails) {
			const d = r.exceptionDetails;
			throw new Error(`page eval failed: ${d.exception?.description ?? d.text}`);
		}
		return r.result?.value as T;
	}

	/**
	 * The composited page (DOM, SVG and WebGL together) over the CSS rectangle 0,0,width,height. `fast` trades file
	 * size for encoding time (for PNG still lossless).
	 */
	async screenshot(width: number, height: number, format: 'png' | 'jpeg' = 'png', quality?: number, fast = true): Promise<Uint8Array> {
		const r = await this.send('Page.captureScreenshot', {
			format,
			...(format === 'jpeg' && quality !== undefined ? {quality} : {}),
			clip: {x: 0, y: 0, width, height, scale: 1},
			fromSurface: true,
			captureBeyondViewport: false,
			optimizeForSpeed: fast,
		});
		return Buffer.from(r.data as string, 'base64');
	}

	async close(): Promise<void> {
		await this.chrome.send('Target.closeTarget', {targetId: this.targetId});
	}
}

async function readDevtoolsUrl(stderr: ReadableStream<Uint8Array>): Promise<string> {
	const reader = stderr.getReader();
	const dec = new TextDecoder();
	let buf = '';
	const deadline = Date.now() + 20000;
	while (Date.now() < deadline) {
		const {value, done} = await reader.read();
		if (done) break;
		buf += dec.decode(value);
		const m = /DevTools listening on (ws:\/\/\S+)/.exec(buf);
		if (m) {
			// keep draining stderr so Chrome never blocks on a full pipe
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

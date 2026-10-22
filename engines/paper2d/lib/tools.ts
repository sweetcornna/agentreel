// 外部程序的位置：Chrome、ffmpeg、ffprobe。环境变量优先（CHROME_PATH / FFMPEG / FFPROBE），其次各平台的常见安装位置，
// 最后在 PATH 里找。找不到时报一行说明怎么设，不猜别的路径。GL 后端的平台默认值也在这里（chrome.ts 里是各后端的参数）。
import {existsSync} from 'node:fs';
import {homedir} from 'node:os';
import {join} from 'node:path';

const which = (name: string): string | null => Bun.which(name);

const firstExisting = (paths: (string | undefined | null)[]): string | null => {
	for (const p of paths) if (p && existsSync(p)) return p;
	return null;
};

/** 平台上 Chrome / Chromium 的常见位置（只是候选，不存在就跳过）。 */
function chromeCandidates(): string[] {
	if (process.platform === 'darwin') {
		const apps = ['Google Chrome.app/Contents/MacOS/Google Chrome', 'Chromium.app/Contents/MacOS/Chromium', 'Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'];
		return [...apps.map((a) => join('/Applications', a)), ...apps.map((a) => join(homedir(), 'Applications', a))];
	}
	if (process.platform === 'win32') {
		const roots = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA].filter(Boolean) as string[];
		return roots.flatMap((r) => [join(r, 'Google', 'Chrome', 'Application', 'chrome.exe'), join(r, 'Chromium', 'Application', 'chrome.exe')]);
	}
	return ['/usr/bin/google-chrome-stable', '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/snap/bin/chromium'];
}

let chromeCache: string | null = null;
/** 无头渲染用的 Chrome：CHROME_PATH → 平台默认位置 → PATH（google-chrome-stable / google-chrome / chromium / chrome）。 */
export function chromePath(): string {
	if (chromeCache) return chromeCache;
	const env = process.env.CHROME_PATH;
	if (env) {
		if (!existsSync(env)) throw new Error(`CHROME_PATH=${env} does not exist`);
		return (chromeCache = env);
	}
	const found =
		firstExisting(chromeCandidates()) ??
		['google-chrome-stable', 'google-chrome', 'chromium', 'chromium-browser', 'chrome'].map(which).find(Boolean) ??
		null;
	if (!found) throw new Error('Google Chrome not found: install it or set CHROME_PATH to the browser executable');
	return (chromeCache = found);
}

const toolCache = new Map<string, string>();
/** ffmpeg / ffprobe：环境变量（FFMPEG / FFPROBE）→ PATH。 */
function tool(name: 'ffmpeg' | 'ffprobe', envName: 'FFMPEG' | 'FFPROBE'): string {
	const hit = toolCache.get(name);
	if (hit) return hit;
	const env = process.env[envName];
	if (env) {
		const p = existsSync(env) ? env : which(env);
		if (!p) throw new Error(`${envName}=${env} does not exist`);
		toolCache.set(name, p);
		return p;
	}
	const p = which(name);
	if (!p) throw new Error(`${name} not found on PATH: install it or set ${envName} to the executable`);
	toolCache.set(name, p);
	return p;
}

export const ffmpeg = (): string => tool('ffmpeg', 'FFMPEG');
export const ffprobe = (): string => tool('ffprobe', 'FFPROBE');

/**
 * GL 后端的平台默认：macOS 用 ANGLE 的 Metal 后端（WebGL2 在 GPU 上），Windows 用 ANGLE 的 D3D11，Linux 默认
 * SwiftShader（无头服务器多半没有 GPU；有 GPU 时传 --gl vulkan 或 egl）。PAPER2D_GL 环境变量或 --gl 覆盖。
 */
export function defaultGl(): string {
	if (process.env.PAPER2D_GL) return process.env.PAPER2D_GL;
	if (process.platform === 'darwin') return 'metal';
	if (process.platform === 'win32') return 'd3d11';
	return 'swangle';
}

/** Run a program; resolve to its stdout, or reject with its stderr. */
export async function run(cmd: string[], input?: Uint8Array): Promise<Uint8Array> {
	const p = Bun.spawn(cmd, {stdin: input ? 'pipe' : 'ignore', stdout: 'pipe', stderr: 'pipe'});
	if (input) {
		const sink = p.stdin as import('bun').FileSink;
		sink.write(input);
		await sink.end();
	}
	const [out, err, code] = await Promise.all([new Response(p.stdout).arrayBuffer(), new Response(p.stderr).text(), p.exited]);
	if (code !== 0) throw new Error(`${cmd.slice(0, 3).join(' ')} … failed (${code}): ${err}`);
	return new Uint8Array(out);
}

/** ffprobe of the first video (or audio) stream as JSON. */
export async function probeStream(file: string, kind: 'v' | 'a', entries: string): Promise<Record<string, any> | null> {
	const out = await run([ffprobe(), '-v', 'error', '-select_streams', `${kind}:0`, '-show_entries', `stream=${entries}`, '-of', 'json', file]);
	return JSON.parse(new TextDecoder().decode(out)).streams?.[0] ?? null;
}

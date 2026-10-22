// Local HTTP server for the render page: the page HTML and the in-memory bundle, plus the film's public/ directory
// served in place under /public (never copied; it can be large). Bound to 127.0.0.1 on a free port.
import {existsSync, readdirSync, statSync} from 'node:fs';
import {join, relative, resolve, sep} from 'node:path';
import type {Bundle} from './build';

export type RunningServer = {url: string; stop(): void};

export const STATIC_BASE = '/public';

type StaticFile = {name: string; src: string; sizeInBytes: number; lastModified: number};

/** What getStaticFiles() returns: every file under public/ (dot files skipped). */
function listPublic(dir: string | null): StaticFile[] {
	const out: StaticFile[] = [];
	if (!dir || !existsSync(dir)) return out;
	const walk = (d: string) => {
		for (const e of readdirSync(d, {withFileTypes: true})) {
			if (e.name.startsWith('.')) continue;
			const p = join(d, e.name);
			if (e.isDirectory()) walk(p);
			else if (e.isFile()) {
				const st = statSync(p);
				const name = relative(dir, p).split(sep).join('/');
				out.push({
					name,
					src: `${STATIC_BASE}/${name.split('/').map(encodeURIComponent).join('/')}`,
					sizeInBytes: st.size,
					lastModified: Math.floor(st.mtimeMs),
				});
			}
		}
	};
	walk(dir);
	return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

const page = (entry: string, injected: unknown) => `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>paper2d render</title>
<style>
/* the page CSS the scenes are designed under: border-box everywhere, no body margin. Without the first rule a padded
   box with a fixed height grows (in the source film a card's third line fell out of the card). */
* { box-sizing: border-box; }
body { margin: 0; }
</style>
</head>
<body>
<div id="paper2d-canvas" style="display:flex;background-color:transparent"></div>
<script>window.__PAPER2D__ = ${JSON.stringify(injected).replace(/</g, '\\u003c')};</script>
<script type="module" src="/${entry}"></script>
</body>
</html>
`;

export function startServer(
	bundle: Bundle,
	o: {publicDir: string | null; inputProps?: Record<string, unknown>; timeout?: number},
): RunningServer {
	const publicDir = o.publicDir ? resolve(o.publicDir) : null;
	const html = page(bundle.entry, {
		inputProps: o.inputProps ?? {},
		staticFiles: listPublic(publicDir),
		staticBase: STATIC_BASE,
		timeout: o.timeout,
	});
	const server = Bun.serve({
		port: 0,
		hostname: '127.0.0.1',
		fetch(req) {
			const url = new URL(req.url);
			let path: string;
			try {
				path = decodeURIComponent(url.pathname);
			} catch {
				return new Response('bad path', {status: 400});
			}
			if (path === '/' || path === '/index.html')
				return new Response(html, {headers: {'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store'}});
			if (path.startsWith(`${STATIC_BASE}/`)) {
				if (!publicDir) return new Response('this film has no public/ directory', {status: 404});
				const rel = path.slice(STATIC_BASE.length + 1);
				const file = resolve(publicDir, rel);
				if (!file.startsWith(publicDir + sep)) return new Response('forbidden', {status: 403});
				if (!existsSync(file) || !statSync(file).isFile()) return new Response('not found', {status: 404});
				return new Response(Bun.file(file));
			}
			const blob = bundle.files.get(path.slice(1));
			if (blob) return new Response(blob, {headers: {'content-type': 'text/javascript; charset=utf-8', 'cache-control': 'no-store'}});
			return new Response('not found', {status: 404});
		},
	});
	return {url: `http://127.0.0.1:${server.port}`, stop: () => server.stop(true)};
}

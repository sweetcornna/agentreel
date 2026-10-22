// Bundles the render page (page/entry.tsx, which imports the film's entry) with Bun, in memory. Nothing is written to
// disk, so concurrent sessions (several agents rendering stills of the same checkout) never share or clobber a bundle.
//
// Module aliases (the film's code needs no install of its own):
//   'remotion', 'paper2d'   -> shim/remotion.ts (one file, so one module instance and one set of React contexts)
//   'paper2d/components'    -> components/index.ts
//   'paper2d:film'          -> the film's entry (src/index.ts of --film, or --entry)
//   'react', 'react-dom', 'react/jsx-runtime', 'scheduler', …  -> this engine's node_modules (exactly one React)
//   '@remotion/*'           -> an error: only the 'remotion' module is provided
import {resolve} from 'node:path';

export const ENGINE_DIR = resolve(import.meta.dir, '..');
const ENTRY = resolve(ENGINE_DIR, 'page/entry.tsx');
export const SHIM = resolve(ENGINE_DIR, 'shim/remotion.ts');
export const COMPONENTS = resolve(ENGINE_DIR, 'components/index.ts');

export type Bundle = {files: Map<string, Blob>; entry: string; ms: number};

export async function buildBundle(filmEntry: string): Promise<Bundle> {
	const t0 = performance.now();
	let result: Awaited<ReturnType<typeof Bun.build>>;
	try {
		result = await Bun.build({
			entrypoints: [ENTRY],
			target: 'browser',
			format: 'esm',
			minify: false,
			sourcemap: 'inline',
			define: {'process.env.NODE_ENV': '"production"'},
			plugins: [
				{
					name: 'paper2d-aliases',
					setup(b) {
						b.onResolve({filter: /^(remotion|paper2d)$/}, () => ({path: SHIM}));
						b.onResolve({filter: /^paper2d\/components$/}, () => ({path: COMPONENTS}));
						b.onResolve({filter: /^paper2d:film$/}, () => ({path: filmEntry}));
						b.onResolve({filter: /^(react|react-dom|scheduler)(\/.*)?$/}, (args) => ({path: Bun.resolveSync(args.path, ENGINE_DIR)}));
						b.onResolve({filter: /^@remotion\//}, (args) => {
							throw new Error(`${args.importer} imports ${args.path}: only 'remotion' (the paper2d shim) is provided`);
						});
					},
				},
			],
		});
	} catch (e) {
		// Bun.build throws an AggregateError whose message is only "Bundle failed"; the reasons are in .errors
		const errs = e instanceof AggregateError ? e.errors.map((x) => (x instanceof Error ? x.message : String(x))) : [e instanceof Error ? e.message : String(e)];
		throw new Error(`bundle failed:\n${errs.join('\n')}`);
	}
	if (!result.success) throw new Error(`bundle failed:\n${result.logs.map(String).join('\n')}`);
	const files = new Map<string, Blob>();
	let entry = '';
	for (const out of result.outputs) {
		const name = out.path.replace(/^\.?\//, '');
		files.set(name, out);
		if (out.kind === 'entry-point') entry = name;
	}
	if (!entry) throw new Error('bundle produced no entry point');
	return {files, entry, ms: performance.now() - t0};
}

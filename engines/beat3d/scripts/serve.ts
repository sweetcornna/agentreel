// Local preview: scrub any frame in the browser (replaces a "studio").
//   bun scripts/serve.ts [--film <dir>] [--port 5173]
// film.js is rebuilt on every page load, so edit -> reload.
import { selectionLine } from './lib/film';
import { parseArgs } from './lib/session';
import { startServer } from './lib/server';

const args = parseArgs();
const s = startServer({ port: Number(args.port ?? 5173), rebuildOnRequest: true });
console.log(selectionLine());
console.log(`viewer: ${s.url}/viewer.html   (←/→ frame, shift = beat, space = play, ?f=<frame>)`);

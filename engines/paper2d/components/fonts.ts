// 字体登记：把片目录 public/ 里的字体文件以 FontFace 登记进页面，加载完之前用 delayRender 挡住帧就绪，
// 所以第一帧就用对字体（不会先用回退字体排一遍）。每个渲染标签页只登记一次。
import {continueRender, delayRender, staticFile} from '../shim/remotion';

export type FontFaceSpec = {
	family: string;
	/** path inside public/, e.g. 'fonts/NotoSerifSC-500-subset.woff2' */
	file: string;
	weight?: string;
	style?: 'normal' | 'italic';
};

const requested = new Set<string>();

/** Register the faces (call once at module level of the film, before any frame renders). */
export function loadFonts(faces: FontFaceSpec[]): void {
	if (typeof document === 'undefined') return;
	const todo = faces.filter((f) => !requested.has(`${f.family}|${f.weight ?? '400'}|${f.style ?? 'normal'}|${f.file}`));
	if (!todo.length) return;
	for (const f of todo) requested.add(`${f.family}|${f.weight ?? '400'}|${f.style ?? 'normal'}|${f.file}`);
	const h = delayRender('fonts', {timeoutInMilliseconds: 120000});
	Promise.all(
		todo.map((f) =>
			new FontFace(f.family, `url(${staticFile(f.file)})`, {weight: f.weight ?? '400', style: f.style ?? 'normal'}).load().then((face) => {
				document.fonts.add(face);
			}),
		),
	)
		.then(() => continueRender(h))
		.catch((e) => {
			// a missing font must not pass silently: the frame would be set in a fallback face
			console.error(`font load failed: ${e instanceof Error ? e.message : String(e)}`);
			continueRender(h);
		});
}

// Render page runtime. Loads the film (its entry registers the root through the shim; lib/build.ts maps the
// 'paper2d:film' import below to it), then exposes window.paper2d for the CDP driver:
//   paper2d.compositions()  -> every <Composition> of the root (id, size, fps, duration, default props)
//   paper2d.select(id)      -> sizes the canvas for that composition; props = defaultProps + injected input props
//   paper2d.frame(n)        -> renders frame n and resolves once it is safe to screenshot (see settle())
import React from 'react';
import {flushSync} from 'react-dom';
import {createRoot, type Root} from 'react-dom/client';
import {AbsoluteFill, RegistryContext, TimelineContext, VideoConfigContext, type CompositionInfo, type VideoConfig} from '../shim/core';
import {injectedInputProps, render, waitForHandles} from '../shim/state';
import 'paper2d:film';

type CompMeta = Omit<CompositionInfo, 'component'>;
type FrameReport = {frame: number; ms: number; rounds: number};

const meta = ({component: _c, ...m}: CompositionInfo): CompMeta => m;

let registry: Map<string, CompositionInfo> | null = null;
const collect = (): Map<string, CompositionInfo> => {
	if (registry) return registry;
	const Root = render.root;
	if (!Root) throw new Error('the film entry never called registerRoot()');
	const found = new Map<string, CompositionInfo>();
	const host = document.createElement('div');
	const root = createRoot(host);
	flushSync(() =>
		root.render(
			<RegistryContext.Provider value={found}>
				<Root />
			</RegistryContext.Provider>,
		),
	);
	root.unmount();
	registry = found;
	return found;
};

let stage: {root: Root; comp: CompositionInfo; config: VideoConfig; portal: HTMLDivElement} | null = null;

const raf = () => new Promise<void>((res) => requestAnimationFrame(() => res()));

/** Every <img> in the document loaded and decoded (the <Img> component already holds a handle for its own). */
const decodeImages = () =>
	Promise.all(
		Array.from(document.images).map((img) =>
			img.complete
				? img.decode().catch(() => undefined)
				: new Promise<void>((res) => {
						img.addEventListener('load', () => res(), {once: true});
						img.addEventListener('error', () => res(), {once: true});
					}).then(() => img.decode().catch(() => undefined)),
		),
	);

/**
 * A frame is ready when React has committed, no delayRender() handle is open, fonts are ready, images are decoded,
 * and two animation frames have passed (WebGL canvases drawn in layout effects have been presented). If a new handle
 * was opened meanwhile (state updates after a load, layout-effect probes), the whole check runs again.
 */
const settle = async (): Promise<number> => {
	let rounds = 0;
	for (;;) {
		rounds++;
		await waitForHandles();
		const generation = render.generation;
		await document.fonts.ready;
		await decodeImages();
		await raf();
		await raf();
		if (render.cancelled) throw new Error(render.cancelled);
		if (render.handles.size === 0 && render.generation === generation) return rounds;
		if (rounds > 50) throw new Error('frame did not settle after 50 rounds (a component keeps opening delayRender handles)');
	}
};

const Stage: React.FC<{comp: CompositionInfo; config: VideoConfig; frame: number}> = ({comp, config, frame}) => {
	const C = comp.component;
	return (
		<VideoConfigContext.Provider value={config}>
			<TimelineContext.Provider value={frame}>
				{/* the composition's own fill: a composition is clipped to its frame */}
				<AbsoluteFill style={{display: 'flex', flexDirection: 'column'}}>
					<C {...config.props} />
				</AbsoluteFill>
			</TimelineContext.Provider>
		</VideoConfigContext.Provider>
	);
};

const paper2d = {
	compositions(): CompMeta[] {
		return [...collect().values()].map(meta);
	},
	select(id: string): CompMeta {
		if (stage) throw new Error('a composition is already selected in this page');
		const comp = collect().get(id);
		if (!comp) throw new Error(`no composition "${id}"; have: ${[...collect().keys()].join(', ')}`);
		const props = {...comp.defaultProps, ...injectedInputProps()};
		const config: VideoConfig = {
			id: comp.id,
			width: comp.width,
			height: comp.height,
			fps: comp.fps,
			durationInFrames: comp.durationInFrames,
			defaultProps: comp.defaultProps,
			props,
			defaultCodec: null,
			defaultOutName: null,
			defaultVideoImageFormat: null,
			defaultPixelFormat: null,
		};
		const canvas = document.getElementById('paper2d-canvas')!;
		canvas.style.width = `${comp.width}px`;
		canvas.style.height = `${comp.height}px`;
		const portal = document.createElement('div');
		Object.assign(portal.style, {
			position: 'absolute',
			top: '0px',
			left: '0px',
			right: '0px',
			bottom: '0px',
			width: '100%',
			height: '100%',
			display: 'flex',
			flexDirection: 'column',
		});
		canvas.appendChild(portal);
		const root = createRoot(portal, {
			onUncaughtError: (err) => {
				const e = err as Error;
				render.cancelled = `uncaught error while rendering: ${e?.stack ?? String(err)}`;
				console.error(render.cancelled);
			},
		});
		stage = {root, comp, config, portal};
		return meta(comp);
	},
	async frame(n: number): Promise<FrameReport> {
		if (!stage) throw new Error('paper2d.select(id) first');
		const {root, comp, config, portal} = stage;
		if (!Number.isInteger(n) || n < 0 || n >= comp.durationInFrames)
			throw new Error(`frame ${n} is outside composition "${comp.id}" (0–${comp.durationInFrames - 1})`);
		const t0 = performance.now();
		// Every frame starts from fresh compositor layers. Chrome keeps per-layer raster state (scale / translation
		// hysteresis) from earlier frames, so the same DOM rasterised after other frames can differ: moving caption
		// glyphs, up to 92/255 on a few thousand pixels (measured on the source film). Two committed frames with the
		// composition hidden drop those layers; with it, a frame's pixels no longer depend on what the tab rendered
		// before (still.ts == render.ts with any number of tabs).
		portal.style.display = 'none';
		await raf();
		await raf();
		await new Promise<void>((res) => setTimeout(res, 0));
		portal.style.display = 'flex';
		flushSync(() => root.render(<Stage comp={comp} config={config} frame={n} />));
		const rounds = await settle();
		return {frame: n, ms: performance.now() - t0, rounds};
	},
};

(window as unknown as {paper2d: typeof paper2d}).paper2d = paper2d;

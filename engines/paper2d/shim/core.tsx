// Timeline and layout part of the shim: the timeline and video-config contexts, Sequence, Freeze, AbsoluteFill, Img,
// Audio, Composition and the hooks.
//
// Frame arithmetic follows the documented model exactly, floating point included: the timeline context holds an
// absolute frame T; every Sequence records its offset (the offset of its parent + its own from); a component sees
// T − offset. A Freeze sets T to frame + offset of the sequence it sits in, so its children see (frame + offset) −
// offset. That is `frame` mathematically, but not always bit for bit (217.6 + 320 − 320 = 217.60000000000002), and
// anything seeded with the frame as a string (components/grain.tsx: rnd('gx' + frame)) depends on that exact value,
// so the arithmetic is kept as described rather than simplified.
import React, {
	createContext,
	forwardRef,
	useContext,
	useImperativeHandle,
	useLayoutEffect,
	useMemo,
	useRef,
} from 'react';
import {cancelRender, continueRender, delayRender} from './state';

// ---------------------------------------------------------------- contexts
export type VideoConfig = {
	id: string;
	width: number;
	height: number;
	fps: number;
	durationInFrames: number;
	defaultProps: Record<string, unknown>;
	props: Record<string, unknown>;
	defaultCodec: null;
	defaultOutName: null;
	defaultVideoImageFormat: null;
	defaultPixelFormat: null;
};

export type CompositionInfo = {
	id: string;
	component: React.ComponentType<Record<string, unknown>>;
	width: number;
	height: number;
	fps: number;
	durationInFrames: number;
	defaultProps: Record<string, unknown>;
};

type SequenceInfo = {
	/** offset of the parent sequence (its parentOffset + from); 0 at the top */
	parentOffset: number;
	/** this sequence's `from` */
	from: number;
	durationInFrames: number;
	width?: number;
	height?: number;
};

/** Absolute timeline frame (the page runtime provides the composition frame; a Freeze replaces it). */
export const TimelineContext = createContext<number | null>(null);
export const VideoConfigContext = createContext<VideoConfig | null>(null);
const SequenceContext = createContext<SequenceInfo | null>(null);
/** Set while the page runtime renders the registered root to collect its <Composition>s. */
export const RegistryContext = createContext<Map<string, CompositionInfo> | null>(null);

const offsetOf = (seq: SequenceInfo | null) => (seq ? seq.parentOffset + seq.from : 0);

const useTimeline = (): number => {
	const t = useContext(TimelineContext);
	if (t === null) throw new Error('useCurrentFrame() outside a composition: call it in a component rendered by a <Composition>');
	return t;
};

// ---------------------------------------------------------------- hooks
export const useCurrentFrame = (): number => {
	const t = useTimeline();
	const seq = useContext(SequenceContext);
	return t - offsetOf(seq);
};

export const useVideoConfig = (): VideoConfig => {
	const config = useContext(VideoConfigContext);
	const seq = useContext(SequenceContext);
	if (!config) throw new Error('useVideoConfig() outside a composition: call it in a component rendered by a <Composition>');
	return useMemo(
		() =>
			seq
				? {...config, durationInFrames: seq.durationInFrames, width: seq.width ?? config.width, height: seq.height ?? config.height}
				: config,
		[config, seq],
	);
};

// ---------------------------------------------------------------- AbsoluteFill
export const AbsoluteFill = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({style, ...other}, ref) => {
	const actual = useMemo(
		(): React.CSSProperties => ({
			position: 'absolute',
			top: 0,
			left: 0,
			right: 0,
			bottom: 0,
			width: '100%',
			height: '100%',
			display: 'flex',
			flexDirection: 'column',
			...style,
		}),
		[style],
	);
	return <div ref={ref} style={actual} {...other} />;
});
AbsoluteFill.displayName = 'AbsoluteFill';

// ---------------------------------------------------------------- Sequence
export type SequenceProps = {
	children?: React.ReactNode;
	from?: number;
	durationInFrames?: number;
	name?: string;
	layout?: 'absolute-fill' | 'none';
	style?: React.CSSProperties;
	className?: string;
	width?: number;
	height?: number;
	showInTimeline?: boolean;
};

export const Sequence = forwardRef<HTMLDivElement, SequenceProps>(
	({children, from = 0, durationInFrames = Infinity, layout = 'absolute-fill', style, className, width, height}, ref) => {
		const t = useTimeline();
		const parent = useContext(SequenceContext);
		if (typeof from !== 'number' || Number.isNaN(from) || !Number.isFinite(from))
			throw new TypeError(`<Sequence from>: must be a finite number, got ${from}`);
		if (typeof durationInFrames !== 'number' || Number.isNaN(durationInFrames) || durationInFrames <= 0)
			throw new TypeError(`<Sequence durationInFrames>: must be positive, got ${durationInFrames}`);
		if (layout !== 'absolute-fill' && layout !== 'none') throw new TypeError(`<Sequence layout>: "absolute-fill" or "none", got ${String(layout)}`);
		if (layout === 'none' && typeof style !== 'undefined') throw new TypeError('<Sequence layout="none"> renders no element, so it takes no style');
		const parentOffset = offsetOf(parent);
		// a sequence ends no later than its parent
		const duration = Math.max(0, Math.min(durationInFrames, parent ? parent.durationInFrames - from : Infinity));
		const info = useMemo(
			(): SequenceInfo => ({parentOffset, from, durationInFrames: duration, width: width ?? parent?.width, height: height ?? parent?.height}),
			[parentOffset, from, duration, width, height, parent],
		);
		const fillStyle = useMemo(
			(): React.CSSProperties => ({flexDirection: undefined, ...(width ? {width} : {}), ...(height ? {height} : {}), ...style}),
			[width, height, style],
		);
		const start = parentOffset + from;
		// children exist only on [start, start + duration); fractional frames (a Retimed picture sampling another grid) included
		if (t < start || t >= start + duration) return null;
		const content = <SequenceContext.Provider value={info}>{children}</SequenceContext.Provider>;
		if (layout === 'none') return content;
		return (
			<AbsoluteFill ref={ref} style={fillStyle} className={className}>
				{content}
			</AbsoluteFill>
		);
	},
);
Sequence.displayName = 'Sequence';

// ---------------------------------------------------------------- Freeze
export const Freeze: React.FC<{frame: number; children?: React.ReactNode; active?: boolean | ((f: number) => boolean)}> = ({
	frame,
	children,
	active = true,
}) => {
	const current = useCurrentFrame();
	const seq = useContext(SequenceContext);
	if (typeof frame === 'undefined') throw new Error('<Freeze>: the frame prop is required');
	if (typeof frame !== 'number') throw new Error(`<Freeze frame>: must be a number, got ${typeof frame}`);
	if (Number.isNaN(frame)) throw new Error('<Freeze frame>: got NaN');
	if (!Number.isFinite(frame)) throw new Error(`<Freeze frame>: must be finite, got ${frame}`);
	const isActive = typeof active === 'function' ? active(current) : active;
	if (!isActive) return <>{children}</>;
	// children see `frame` (possibly fractional) relative to the sequence the Freeze sits in; Sequences inside count
	// from it. The timeline value is frame + offset, so children compute (frame + offset) − offset (see the header).
	return <TimelineContext.Provider value={frame + offsetOf(seq)}>{children}</TimelineContext.Provider>;
};

// ---------------------------------------------------------------- Img: holds the frame until the picture is decoded
export type ImgProps = Omit<React.ImgHTMLAttributes<HTMLImageElement>, 'src'> & {
	src: string;
	maxRetries?: number;
	pauseWhenLoading?: boolean;
	delayRenderRetries?: number;
	delayRenderTimeoutInMilliseconds?: number;
	onImageFrame?: (img: HTMLImageElement) => void;
};

export const Img = forwardRef<HTMLImageElement, ImgProps>(
	({src, onError, maxRetries: _maxRetries, pauseWhenLoading: _pause, delayRenderRetries: _retries, delayRenderTimeoutInMilliseconds, onImageFrame, ...props}, ref) => {
		const imageRef = useRef<HTMLImageElement>(null);
		useImperativeHandle(ref, () => imageRef.current as HTMLImageElement, []);
		if (!src) throw new Error('<Img>: the src prop is required');
		useLayoutEffect(() => {
			const el = imageRef.current;
			if (!el) return;
			const handle = delayRender(`Loading <Img> with src=${src}`, {timeoutInMilliseconds: delayRenderTimeoutInMilliseconds});
			let released = false;
			const release = () => {
				if (released) return;
				released = true;
				onImageFrame?.(el);
				continueRender(handle);
			};
			el.decode().then(release, () => {
				if (el.complete && el.naturalWidth > 0) release();
				else el.addEventListener('load', release, {once: true});
			});
			return () => {
				el.removeEventListener('load', release);
				if (!released) {
					released = true;
					continueRender(handle);
				}
			};
		}, [src, delayRenderTimeoutInMilliseconds, onImageFrame]);
		const didError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
			if (onError) onError(e);
			else cancelRender(new Error(`<Img> could not load ${src}`));
		};
		return <img {...props} ref={imageRef} src={src} decoding="sync" onError={didError} />;
	},
);
Img.displayName = 'Img';

// ---------------------------------------------------------------- Audio: pictures only; the score is muxed separately (mux.ts)
export const Audio: React.FC<Record<string, unknown>> = () => null;

// ---------------------------------------------------------------- Composition / Still / Folder
export type CompositionProps<P extends Record<string, unknown>> = {
	id: string;
	component?: React.ComponentType<P>;
	lazyComponent?: () => Promise<{default: React.ComponentType<P>}>;
	durationInFrames: number;
	fps: number;
	width: number;
	height: number;
	defaultProps?: P;
	schema?: unknown;
	calculateMetadata?: unknown;
};

const positiveInt = (v: unknown, what: string, id: string) => {
	if (typeof v !== 'number' || !Number.isInteger(v) || v <= 0) throw new TypeError(`<Composition id="${id}" ${what}>: must be a positive integer, got ${String(v)}`);
};

export const Composition = <P extends Record<string, unknown>>(props: CompositionProps<P>): null => {
	const registry = useContext(RegistryContext);
	const {id, component, lazyComponent, durationInFrames, fps, width, height, defaultProps} = props;
	if (typeof id !== 'string' || !/^[a-zA-Z0-9一-鿿-]+$/.test(id))
		throw new Error(`<Composition id>: letters, digits, CJK characters and - only, got ${String(id)}`);
	if (lazyComponent) throw new Error(`<Composition id="${id}">: lazyComponent is not supported here; pass component`);
	if (!component) throw new Error(`<Composition id="${id}">: no component`);
	positiveInt(durationInFrames, 'durationInFrames', id);
	positiveInt(width, 'width', id);
	positiveInt(height, 'height', id);
	if (typeof fps !== 'number' || !Number.isFinite(fps) || fps <= 0) throw new TypeError(`<Composition id="${id}" fps>: must be positive, got ${fps}`);
	if (registry) {
		if (registry.has(id)) throw new Error(`<Composition id="${id}">: registered twice`);
		registry.set(id, {
			id,
			component: component as unknown as React.ComponentType<Record<string, unknown>>,
			width,
			height,
			fps,
			durationInFrames,
			defaultProps: (defaultProps ?? {}) as Record<string, unknown>,
		});
	}
	return null;
};

export const Still = <P extends Record<string, unknown>>(props: Omit<CompositionProps<P>, 'durationInFrames' | 'fps'>) => (
	<Composition {...(props as CompositionProps<P>)} durationInFrames={1} fps={1} />
);

export const Folder: React.FC<{name: string; children?: React.ReactNode}> = ({children}) => <>{children}</>;

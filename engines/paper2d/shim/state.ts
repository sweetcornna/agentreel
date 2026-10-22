// Page-wide render state behind the shim: delayRender handles, cancellation, the registered root, and what the
// render page injects (window.__PAPER2D__: input props, the public/ file list, the static base). The page runtime
// (page/entry.tsx) reads this to decide when a frame is ready.
import type React from 'react';

export type StaticFile = {name: string; src: string; sizeInBytes: number; lastModified: number};

type Injected = {
	inputProps?: Record<string, unknown>;
	staticFiles?: StaticFile[];
	staticBase?: string;
	/** default delayRender() timeout, ms (30 s unless --timeout says otherwise) */
	timeout?: number;
};

const injected = (): Injected =>
	typeof window === 'undefined' ? {} : ((window as unknown as {__PAPER2D__?: Injected}).__PAPER2D__ ?? {});

export const STATIC_BASE = () => injected().staticBase ?? '/public';
export const injectedInputProps = (): Record<string, unknown> => ({...(injected().inputProps ?? {})});
export const injectedStaticFiles = (): StaticFile[] => injected().staticFiles ?? [];

type Handle = {label: string; started: number; timeout: number};

export const render = {
	handles: new Map<number, Handle>(),
	/** bumped by every delayRender(): a frame is settled only if no new handle appeared while waiting */
	generation: 0,
	cancelled: null as string | null,
	root: null as React.ComponentType | null,
	waiters: [] as (() => void)[],
};

let nextHandle = 1;

const wake = () => {
	const w = render.waiters.splice(0);
	for (const f of w) f();
};

export const delayRender = (label?: string, options?: {timeoutInMilliseconds?: number; retries?: number}): number => {
	if (typeof label !== 'undefined' && typeof label !== 'string')
		throw new Error(`delayRender(label): label must be a string or undefined, got ${JSON.stringify(label)}`);
	const handle = nextHandle++;
	render.handles.set(handle, {
		label: label ?? 'delayRender()',
		started: typeof performance === 'undefined' ? Date.now() : performance.now(),
		timeout: options?.timeoutInMilliseconds ?? injected().timeout ?? 30000,
	});
	render.generation++;
	return handle;
};

export const continueRender = (handle: number): void => {
	if (typeof handle === 'undefined') throw new TypeError('continueRender(handle): pass the number delayRender() returned (got nothing)');
	if (typeof handle !== 'number') throw new TypeError(`continueRender(handle): handle must be the number delayRender() returned, got ${JSON.stringify(handle)}`);
	if (!render.handles.delete(handle)) return;
	if (render.handles.size === 0) wake();
};

export const cancelRender = (err: unknown): never => {
	let error: Error;
	if (err instanceof Error) error = err;
	else if (typeof err === 'string') error = new Error(err);
	else error = new Error('render cancelled');
	render.cancelled = error.stack ?? error.message;
	console.error('cancelRender:', render.cancelled);
	wake();
	throw error;
};

/** Resolves when every handle is continued; rejects on cancelRender() or when a handle outlives its timeout. */
export const waitForHandles = async (): Promise<void> => {
	for (;;) {
		if (render.cancelled) throw new Error(render.cancelled);
		if (render.handles.size === 0) return;
		const now = performance.now();
		for (const h of render.handles.values())
			if (now - h.started > h.timeout) throw new Error(`delayRender("${h.label}") was still open after ${h.timeout} ms`);
		await new Promise<void>((res) => {
			render.waiters.push(res);
			setTimeout(res, 50);
		});
	}
};

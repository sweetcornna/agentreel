// The module a film's code sees as 'remotion' (or 'paper2d'): lib/build.ts aliases both bare specifiers to this file.
// It is a compatible subset of the Remotion 4 API, written from its documented behaviour; no Remotion code is used
// and nothing needs to be installed (see README.md for the contract, API by API).
import type React from 'react';
import {injectedInputProps, injectedStaticFiles, render, STATIC_BASE, type StaticFile} from './state';

export {cancelRender, continueRender, delayRender} from './state';
export type {StaticFile} from './state';
export {Easing, interpolate, measureSpring, random, spring} from './math';
export type {EasingFunction, ExtrapolateType, InterpolateOptions, SpringConfig} from './math';
export {AbsoluteFill, Audio, Composition, Folder, Freeze, Img, Sequence, Still, useCurrentFrame, useVideoConfig} from './core';
export type {CompositionProps, ImgProps, SequenceProps, VideoConfig} from './core';

export const registerRoot = (comp: React.ComponentType): void => {
	if (!comp) throw new Error(`registerRoot(): expected a React component, got ${JSON.stringify(comp)}`);
	if (render.root) throw new Error('registerRoot() was called twice; a film registers one root');
	render.root = comp;
};

/** URL of a file in the film's public/ directory, served by lib/server.ts under the static base. */
export const staticFile = (path: string): string => {
	if (path === null) throw new TypeError('staticFile(null)');
	if (typeof path === 'undefined') throw new TypeError('staticFile(undefined)');
	if (path.startsWith('http://') || path.startsWith('https://'))
		throw new TypeError(`staticFile("${path}"): remote URLs are not files in public/; use the URL itself`);
	if (path.startsWith('..') || path.startsWith('./'))
		throw new TypeError(`staticFile("${path}"): give the path inside public/, without ./ or ..`);
	const encoded = path
		.replace(/^\/+/, '')
		.split('/')
		.map((p) => encodeURIComponent(p))
		.join('/');
	return `${STATIC_BASE()}/${encoded}`;
};

export const getStaticFiles = (): StaticFile[] => injectedStaticFiles();

export const getInputProps = (): Record<string, unknown> => injectedInputProps();

// Put the score under a rendered picture.
//   bun mux.ts --video <picture.mp4> --audio <score.wav> [--out <film.mp4>] [--bitrate 320k] [--offset seconds]
// The video stream is copied as it is; the audio is encoded to AAC-LC; the length follows the picture (a shorter score
// is padded with silence, a longer one is cut, both with a warning). Default output: <picture minus .mp4>_audio.mp4.
import {resolve} from 'node:path';
import {muxAudio} from './lib/mux';
import {parseArgs, runCli, UsageError} from './lib/session';

await runCli(async () => {
	const args = parseArgs();
	if (typeof args.video !== 'string' || typeof args.audio !== 'string')
		throw new UsageError('usage: bun mux.ts --video <picture.mp4> --audio <score.wav> [--out film.mp4] [--bitrate 320k] [--offset s]');
	const video = resolve(args.video);
	const out = resolve(typeof args.out === 'string' ? args.out : video.replace(/\.mp4$/i, '') + '_audio.mp4');
	const offset = typeof args.offset === 'string' ? Number(args.offset) : 0;
	if (!Number.isFinite(offset)) throw new UsageError('--offset needs seconds');
	const m = await muxAudio({video, audio: resolve(args.audio), out, bitrate: typeof args.bitrate === 'string' ? args.bitrate : undefined, offset});
	console.log(`picture ${m.videoSeconds.toFixed(3)} s, score ${m.audioSeconds.toFixed(3)} s -> ${m.out}`);
});

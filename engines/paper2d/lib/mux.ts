// 画面 + 配乐 → 带声成片。视频流原样拷贝（不重编码），音频编成 AAC-LC；长度以画面为准：配乐短了用静音补齐，
// 长了截到画面长度并提示（配乐的尾巴应当已经写进 timeline 的 totalFrames，例如示例片的 7.1–8.1 余音）。
import {existsSync, renameSync, unlinkSync} from 'node:fs';
import {ffmpeg, probeStream, run} from './tools';

export type MuxResult = {out: string; videoSeconds: number; audioSeconds: number; args: string[]};

export async function muxAudio(o: {video: string; audio: string; out: string; bitrate?: string; offset?: number}): Promise<MuxResult> {
	if (!existsSync(o.video)) throw new Error(`no picture ${o.video}`);
	if (!existsSync(o.audio)) throw new Error(`no score ${o.audio}`);
	const v = await probeStream(o.video, 'v', 'nb_frames,r_frame_rate,duration');
	const a = await probeStream(o.audio, 'a', 'duration,sample_rate,channels');
	if (!v) throw new Error(`${o.video} has no video stream`);
	if (!a) throw new Error(`${o.audio} has no audio stream`);
	const [fa, fb] = String(v.r_frame_rate).split('/').map(Number);
	const fps = fa / fb;
	const videoSeconds = v.nb_frames ? Number(v.nb_frames) / fps : Number(v.duration);
	const audioSeconds = Number(a.duration);
	const offset = o.offset ?? 0;
	if (audioSeconds + offset > videoSeconds + 0.02)
		console.warn(`[mux] the score (${audioSeconds.toFixed(3)} s) is longer than the picture (${videoSeconds.toFixed(3)} s); cut at the picture's end`);
	if (audioSeconds + offset < videoSeconds - 0.02)
		console.warn(`[mux] the score (${audioSeconds.toFixed(3)} s) is shorter than the picture (${videoSeconds.toFixed(3)} s); padded with silence`);
	const part = `${o.out}.part.mp4`;
	const args = [
		ffmpeg(), '-v', 'error', '-y',
		'-i', o.video,
		...(offset ? ['-itsoffset', offset.toFixed(6)] : []),
		'-i', o.audio,
		'-map', '0:v:0', '-map', '1:a:0',
		'-c:v', 'copy',
		'-af', 'apad',
		'-c:a', 'aac', '-b:a', o.bitrate ?? '320k',
		'-t', videoSeconds.toFixed(6),
		'-movflags', '+faststart',
		part,
	];
	try {
		await run(args);
	} catch (e) {
		if (existsSync(part)) unlinkSync(part);
		throw e;
	}
	renameSync(part, o.out);
	return {out: o.out, videoSeconds, audioSeconds, args};
}

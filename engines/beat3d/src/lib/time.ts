// Musical time <-> frames. Every shot boundary and event is authored as "bar.beat" in the film's
// timeline (films/<name>/timeline.json) and converted here, never hard-coded as a frame number.
//
// Which timeline: the page bundle has it baked in by scripts/build.ts (__TIMELINE__); Bun-side
// scripts get it from scripts/lib/film.ts (--film), which puts it on globalThis before this module
// runs. With neither (an editor, a test) importing this module throws: there is no default film.
//
// Two tempo modes (timeline `tempo`):
//   "fixed" (default): one tempo, framesPerBeat frames per beat. The score's events
//     (films/<name>/out/music/events.json), when they exist, are baked into the page bundle by
//     scripts/build.ts, which checks that their beats sit on this grid within 1 ms; the picture can
//     then follow individual notes (pulses, plucks, ticks). Bun-side scripts run on the grid, which is
//     the same thing to the millisecond.
//   "score": the score's beat times are the clock (a ritardando in the music moves the picture with
//     it). Beats are baked into the bundle and published to Bun-side scripts too; without a score the
//     grid is used.

export type PosObject = { bar: number; beat: number; frac?: number };
/**
 * "5.1", "4.3+0.5" (half a beat after 4.3), "5.1+3f" (3 frames after 5.1), a PosObject, or a raw
 * 0-based beat index (number).
 */
export type Pos = string | number | PosObject;

export interface ShotSpec {
  id: string;
  zh: string;
  from: string;
  to: string;
  /** Free text (or a focal length in mm) describing the lens, camera and content of the shot. */
  lens?: string | number | null;
  camera?: string;
  content?: string;
  owner?: string;
  continuesInto?: string;
  /** Card id this shot carries (informational). */
  card?: string;
  /** Montage cuts inside the shot (qa-cuts expects a jump on each). */
  cuts?: { at: string; look?: string }[];
}
export interface SectionSpec {
  id: string;
  zh: string;
  from: string;
  to: string;
}
/** A sync point. `at` is a position, or a range "1.3-2.4" (its start counts for the cut checks). */
export interface SyncSpec {
  id?: string;
  at: string;
  hard?: boolean;
  picture?: string;
  music?: string;
  event?: string;
}
/** Display-referred cards the engine draws itself unless a program claims them (text/cards.ts). */
export interface CardSpec {
  id: string;
  kind: 'typed' | 'cinema' | 'title' | 'credits';
  zh?: string;
  en?: string;
  from: string;
  to: string;
  tone?: 'warm' | 'cool' | 'neutral';
  source?: { zh: string; en: string };
  [k: string]: unknown;
}
export type CurveKeys = [string, number][];
export interface TimelineSpec {
  fps: number;
  bpm: number;
  beatsPerBar: number;
  framesPerBeat: number;
  framesPerBar: number;
  bars: number;
  totalFrames: number;
  /** "fixed" (default) or "score" (beat times from the score's events.json). */
  tempo?: 'fixed' | 'score';
  canvas: { preview: [number, number]; final: [number, number] };
  /** Visible picture aspect. `interp`: "constant", "linear", "smooth", "easeInOutCubic" (default) ...
   *  (only its first word counts, so a note may follow); `eases`: per-key easing of the segment it starts. */
  aspect: { keys: CurveKeys; interp?: string; eases?: Record<string, string> };
  /** Intention curve 0..1 shared by picture and music (optional). */
  energy?: { keys: CurveKeys };
  /** Trail persistence 0..1 (optional, lib/persistence.ts). */
  persistence?: { keys: CurveKeys };
  /** A guide line the film keeps at one screen height (optional; layout.lineY). */
  line?: { yFrac: number };
  sections?: SectionSpec[];
  shots: ShotSpec[];
  sync?: SyncSpec[];
  cards?: CardSpec[];
}

/** One note or marker of the score, time in seconds from 1.1. */
export interface MusicEvent {
  t: number;
  type: string;
  [k: string]: unknown;
}

declare const __MUSIC_EVENTS__: string | undefined;
declare const __MUSIC_BEATS__: string | undefined;
declare const __TIMELINE__: string | undefined;

interface Host {
  __B3D_TIMELINE__?: unknown;
  __B3D_BEATS__?: number[] | null;
}
const host = globalThis as Host;
const timelineSource: unknown = typeof __TIMELINE__ === 'string' ? JSON.parse(__TIMELINE__) : host.__B3D_TIMELINE__;
if (!timelineSource) throw new Error('beat3d: no timeline. The page gets it from scripts/build.ts; Bun-side scripts must import scripts/lib/film.ts first.');
export const TIMELINE = timelineSource as TimelineSpec;

/** Score events baked in by the build (page only); empty on the Bun side and before the score exists. */
const EVENTS: MusicEvent[] = typeof __MUSIC_EVENTS__ === 'string' ? (JSON.parse(__MUSIC_EVENTS__) as MusicEvent[]) : [];
/** Beat times (s) of a "score" tempo timeline: baked into the page, published by scripts/lib/film.ts on the Bun side. */
const BEATS: number[] | null = typeof __MUSIC_BEATS__ === 'string' ? (JSON.parse(__MUSIC_BEATS__) as number[] | null) : (host.__B3D_BEATS__ ?? null);

export function parsePos(p: Pos): PosObject {
  if (typeof p === 'number') {
    const bpb = TIMELINE.beatsPerBar;
    const whole = Math.floor(p);
    return { bar: Math.floor(whole / bpb) + 1, beat: (whole % bpb) + 1, frac: p - whole };
  }
  if (typeof p === 'object') return { bar: p.bar, beat: p.beat, frac: p.frac ?? 0 };
  const m = /^\s*(\d+)\.(\d+)(?:\s*\+\s*([0-9.]+)(f?))?\s*$/.exec(p);
  if (!m) throw new Error(`Bad position "${p}" (expected "bar.beat", "bar.beat+frac" or "bar.beat+Nf")`);
  const off = m[3] ? Number(m[3]) : 0;
  return { bar: Number(m[1]), beat: Number(m[2]), frac: m[4] === 'f' ? off / TIMELINE.framesPerBeat : off };
}

/** 0-based fractional beat index: "1.1" -> 0, "5.1" -> 16, "4.3+0.5" -> 14.5. */
export function posToBeat(p: Pos): number {
  if (typeof p === 'number') return p;
  const o = parsePos(p);
  return (o.bar - 1) * TIMELINE.beatsPerBar + (o.beat - 1) + (o.frac ?? 0);
}

/** Start of a sync point or range ("1.3-2.4" -> "1.3"). */
export const syncStart = (at: string): string => at.split(/\s*-\s*(?=\d)/)[0];

export class BeatClock {
  readonly fps: number;
  /** Frames per beat on the grid (a "score" tempo timeline may stretch individual beats). */
  readonly framesPerBeat: number;
  /** 'events' when the score's events are baked into this bundle; 'beats' when its beat times drive the clock. */
  readonly source: 'grid' | 'events' | 'beats';
  private readonly beatTimes: number[] | null;

  constructor(
    tl: TimelineSpec,
    private readonly events: MusicEvent[],
    beats: number[] | null,
  ) {
    this.fps = tl.fps;
    this.framesPerBeat = tl.framesPerBeat;
    const valid = tl.tempo === 'score' && Array.isArray(beats) && beats.length >= 2 && beats.every((t, i) => i === 0 || t > beats[i - 1]);
    this.beatTimes = valid ? beats : null;
    this.source = valid ? 'beats' : events.length ? 'events' : 'grid';
  }

  /** Fractional frame of a fractional beat index. */
  beatToFrame(b: number): number {
    const bt = this.beatTimes;
    if (!bt) return b * this.framesPerBeat;
    const n = bt.length;
    if (b <= 0) return (bt[0] + b * (bt[1] - bt[0])) * this.fps;
    if (b >= n - 1) return (bt[n - 1] + (b - (n - 1)) * (bt[n - 1] - bt[n - 2])) * this.fps;
    const i = Math.floor(b);
    return (bt[i] + (b - i) * (bt[i + 1] - bt[i])) * this.fps;
  }
  frameToBeat(f: number): number {
    const bt = this.beatTimes;
    if (!bt) return f / this.framesPerBeat;
    const t = f / this.fps;
    const n = bt.length;
    let b: number;
    if (t <= bt[0]) b = (t - bt[0]) / (bt[1] - bt[0]);
    else if (t >= bt[n - 1]) b = n - 1 + (t - bt[n - 1]) / (bt[n - 1] - bt[n - 2]);
    else {
      let lo = 0;
      let hi = n - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (bt[mid] <= t) lo = mid;
        else hi = mid;
      }
      b = lo + (t - bt[lo]) / (bt[hi] - bt[lo]);
    }
    // events.json rounds beat times to the microsecond, so a frame that lands on a beat can come out a
    // hair before it and anything that floors the beat fires a frame late. Frames are ~0.05 beat apart,
    // so snapping within 1e-4 never moves a real one.
    const r = Math.round(b);
    return Math.abs(b - r) < 1e-4 ? r : b;
  }
  /** Fractional frame of a position. */
  frameAt(p: Pos): number {
    return this.beatToFrame(posToBeat(p));
  }
  /** Integer frame of a position (shot boundaries, stills). */
  f(p: Pos): number {
    return Math.round(this.frameAt(p));
  }
  /** Frames spanned by one beat starting at `p` (framesPerBeat on the grid). */
  beatLength(p: Pos): number {
    const b = posToBeat(p);
    return this.beatToFrame(b + 1) - this.beatToFrame(b);
  }
  /** Phase within the current beat, 0..1. */
  beatPhase(frame: number): number {
    const b = this.frameToBeat(frame);
    return b - Math.floor(b);
  }
  /** "5.3" style label (with "+Nf" if not on a beat). */
  label(frame: number): string {
    const b = Math.floor(this.frameToBeat(frame) + 1e-9);
    const o = parsePos(b);
    const off = Math.round(frame - this.beatToFrame(b));
    return `${o.bar}.${o.beat}${off > 0 ? `+${off}f` : ''}`;
  }

  /** Score events of a type as fractional frames in [fromFrame, toFrame). Empty until the score exists. */
  eventFrames(type: string, fromFrame = -Infinity, toFrame = Infinity): { frame: number; data: MusicEvent }[] {
    return this.events
      .filter((e) => e.type === type)
      .map((e) => ({ frame: e.t * this.fps, data: e }))
      .filter((e) => e.frame >= fromFrame && e.frame < toFrame);
  }
  hasEvents(type: string): boolean {
    return this.events.some((e) => e.type === type);
  }
}

export const CLOCK = new BeatClock(TIMELINE, EVENTS, BEATS);

export interface ShotTiming extends ShotSpec {
  fromFrame: number;
  toFrame: number;
  duration: number;
  index: number;
}

export const SHOTS: ShotTiming[] = TIMELINE.shots.map((s, index) => {
  const fromFrame = CLOCK.f(s.from);
  const toFrame = CLOCK.f(s.to);
  return { ...s, fromFrame, toFrame, duration: toFrame - fromFrame, index };
});

/** Total frames of the timeline, picture and tail (whichever is longer). */
export const FILM_FRAMES = Math.max(TIMELINE.totalFrames, SHOTS[SHOTS.length - 1].toFrame);

/** The shot on screen at a frame, or null where the timeline has no picture (black: the tail). */
export function shotAt(frame: number): ShotTiming | null {
  const f = Math.floor(frame);
  for (const s of SHOTS) if (f >= s.fromFrame && f < s.toFrame) return s;
  return null;
}
export function shotById(id: string): ShotTiming {
  const s = SHOTS.find((x) => x.id === id);
  if (!s) throw new Error(`Unknown shot ${id}`);
  return s;
}
/**
 * A shot's frames, or NaN where this timeline has no such shot. For module-level cues of programs
 * that are registered under several timelines: they load, and are never asked for a frame there.
 */
export function shotSpan(id: string): { fromFrame: number; toFrame: number } {
  const s = SHOTS.find((x) => x.id === id);
  return s ? { fromFrame: s.fromFrame, toFrame: s.toFrame } : { fromFrame: NaN, toFrame: NaN };
}
export function cardById(id: string): CardSpec {
  const c = (TIMELINE.cards ?? []).find((x) => x.id === id);
  if (!c) throw new Error(`Unknown card ${id}`);
  return c;
}

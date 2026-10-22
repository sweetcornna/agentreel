// public/fonts 里的 OFL 字体，用 FontFace API 加载。字一律用 Canvas2D 栅格化成纹理、由 WebGL 画，画面里没有 DOM 字。
//
// 随引擎分发：Noto Sans SC（中文）、Inter（拉丁无衬线）、JetBrains Mono、EB Garamond。Noto Serif SC 是可选的
// （每个字重约 8 MB）：把 NotoSerifSC-300/400/500.woff 放进 public/fonts，构建就会认出来（scripts/build.ts
// 把现有的字体文件列表烘进包）。没有时衬线字族通过 fontCss() 写出的 CSS 字族列表退到无衬线。

export const FONT = {
  serifSC: 'B3D Serif SC',
  sansSC: 'B3D Sans SC',
  sans: 'B3D Sans',
  garamond: 'B3D Garamond',
  mono: 'B3D Mono',
} as const;

/** 某个字族不可用时退到哪个（按 CSS 字族列表的顺序）。 */
const FALLBACK: Record<string, string> = {
  [FONT.serifSC]: FONT.sansSC,
  [FONT.garamond]: FONT.sans,
};

interface FaceSpec {
  family: string;
  file: string;
  weight: string;
  style?: string;
  /** 不随引擎分发：文件在才加载。 */
  optional?: boolean;
}

const FACES: FaceSpec[] = [
  { family: FONT.serifSC, file: 'NotoSerifSC-300.woff', weight: '300', optional: true },
  { family: FONT.serifSC, file: 'NotoSerifSC-400.woff', weight: '400', optional: true },
  { family: FONT.serifSC, file: 'NotoSerifSC-500.woff', weight: '500', optional: true },
  { family: FONT.sansSC, file: 'NotoSansSC-400.woff2', weight: '400' },
  { family: FONT.sansSC, file: 'NotoSansSC-500.woff2', weight: '500' },
  { family: FONT.sans, file: 'Inter-400.woff2', weight: '400' },
  { family: FONT.sans, file: 'Inter-500.woff2', weight: '500' },
  { family: FONT.sans, file: 'Inter-600.woff2', weight: '600' },
  { family: FONT.garamond, file: 'EBGaramond-400.woff', weight: '400' },
  { family: FONT.garamond, file: 'EBGaramond-400i.woff', weight: '400', style: 'italic' },
  { family: FONT.garamond, file: 'EBGaramond-500.woff', weight: '500' },
  { family: FONT.mono, file: 'JetBrainsMono-300.woff', weight: '300' },
  { family: FONT.mono, file: 'JetBrainsMono-400.woff', weight: '400' },
  { family: FONT.mono, file: 'JetBrainsMono-500.woff', weight: '500' },
];

/** public/fonts 里现有的字体文件，由 scripts/build.ts 烘进来（undefined：当作随引擎分发的都在）。 */
declare const __FONT_FILES__: string | undefined;
const PRESENT: Set<string> | null = typeof __FONT_FILES__ === 'string' ? new Set(JSON.parse(__FONT_FILES__) as string[]) : null;

let loading: Promise<void> | null = null;

/** Load every available face once per page. Resolves when all are usable by Canvas2D. */
export function loadFonts(baseUrl = 'fonts/'): Promise<void> {
  if (loading) return loading;
  loading = (async () => {
    const use = FACES.filter((f) => (PRESENT ? PRESENT.has(f.file) : !f.optional));
    const faces = use.map((f) => new FontFace(f.family, `url(${baseUrl}${f.file})`, { weight: f.weight, style: f.style ?? 'normal' }));
    const loaded = await Promise.all(faces.map((f) => f.load()));
    for (const f of loaded) document.fonts.add(f);
    await document.fonts.ready;
  })();
  return loading;
}

/** CSS font shorthand for Canvas2D. `px` is in canvas pixels (already scaled). */
export const fontCss = (family: string, weight: number | string, px: number, style = 'normal'): string =>
  `${style} ${weight} ${px.toFixed(3)}px "${family}"${FALLBACK[family] ? `, "${FALLBACK[family]}"` : ''}`;

/** CJK ideographs, kana, full-width punctuation and CJK symbols. */
export function isCJK(ch: string): boolean {
  const c = ch.codePointAt(0) ?? 0;
  return (
    (c >= 0x3000 && c <= 0x303f) ||
    (c >= 0x3040 && c <= 0x30ff) ||
    (c >= 0x3400 && c <= 0x4dbf) ||
    (c >= 0x4e00 && c <= 0x9fff) ||
    (c >= 0xf900 && c <= 0xfaff) ||
    (c >= 0xff00 && c <= 0xffef)
  );
}

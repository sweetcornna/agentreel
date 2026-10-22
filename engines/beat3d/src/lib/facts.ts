// 片子的事实表（films/<name>/facts.json，可选）：画面上出现的每个数字、名字、每行「真实」文字都从这里取，
// 不在镜头程序里手打，所以每一帧都能对回出处。构建时烘进页面包（__FACTS__）。文件还不存在、或缺某个字段时，
// 这些函数返回醒目的占位 `FACT:<字段>`：画面上永远不会把编出来的字冒充成真的。
//
// 字段可以直接是值，也可以是 { value, source }（文字还可以是 { text | line | code }）；用点号取嵌套字段：
// factText('server.route')。

declare const __FACTS__: string | undefined;

export const FACTS: Record<string, unknown> | null = typeof __FACTS__ === 'string' ? (JSON.parse(__FACTS__) as Record<string, unknown>) : null;

export const placeholder = (field: string): string => `FACT:${field}`;

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string | undefined => (typeof v === 'string' && v.length ? v : undefined);

/** 一个事实的值（拆开 `{ value, source }`），没有就是 undefined。 */
export function factValue(path: string): unknown {
  let v: unknown = FACTS;
  for (const k of path.split('.')) {
    if (v === null || typeof v !== 'object') return undefined;
    v = (v as Record<string, unknown>)[k];
  }
  if (isObj(v) && 'value' in v) v = v.value;
  return v;
}

/** 值的文字：字符串本身，或对象的 text / line / code / raw / value。 */
export function textOf(v: unknown): string | undefined {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  if (isObj(v)) return str(v.text) ?? str(v.line) ?? str(v.code) ?? str(v.raw) ?? str(v.value);
  return undefined;
}

/** 列表：值本身，或对象的 items / list / records / lines。 */
export function listOf(v: unknown): unknown[] | undefined {
  if (Array.isArray(v)) return v;
  if (isObj(v)) for (const k of ['items', 'list', 'records', 'lines']) if (Array.isArray(v[k])) return v[k] as unknown[];
  return undefined;
}

/** 事实的文字，或它的占位；`missing` 说明是哪一种。 */
export function factText(path: string): { text: string; missing: boolean } {
  const t = textOf(factValue(path));
  return t !== undefined ? { text: t, missing: false } : { text: placeholder(path), missing: true };
}

/** 事实作为一组文字（缺的话给 `count` 个占位）。 */
export function factLines(path: string, count = 1): { lines: string[]; missing: boolean } {
  const list = listOf(factValue(path));
  const lines = (list ?? []).map((x) => textOf(x)).filter((x): x is string => x !== undefined);
  if (!lines.length) return { lines: Array.from({ length: count }, (_, i) => placeholder(`${path}[${i}]`)), missing: true };
  return { lines, missing: false };
}

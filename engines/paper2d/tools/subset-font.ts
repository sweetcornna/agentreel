// 字体子集：只留一部片画面上用到的字形，出 woff2。中文字体全量 8 MB 上下，示例片只要几十个字，子集几十 KB，
// 可以放进仓库（OFL 允许修改与再发布；子集是 OFL 意义上的修改版，仍按 OFL 发布，并保留字体里的版权与许可名称记录）。
//   bun tools/subset-font.ts --font <源字体 .otf/.ttf/.woff/.woff2> --out <目标.woff2>
//                            [--text-file <文件>]... [--text "额外的字"] [--no-ascii]
// --text-file 可以给多次：.ts/.tsx/.js/.json 只取字符串字面量里的字（注释不算），其他文件取全部内容。默认再加上
// 可打印 ASCII（空格到 ~）。需要 fonttools：PATH 上的 pyftsubset（PYFTSUBSET 覆盖），否则用 uvx 临时装
// （`uvx --from fonttools[woff] pyftsubset`）。
import {mkdtempSync, readFileSync, rmSync, statSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {extname, join, resolve} from 'node:path';
import {runCli, UsageError} from '../lib/session';

/** Characters inside '…', "…" and `…` literals (escapes kept literally; good enough for display copy). */
const literals = (src: string): string => {
	let out = '';
	const re = /'((?:\\.|[^'\\\n])*)'|"((?:\\.|[^"\\\n])*)"|`((?:\\.|[^`\\])*)`|\/\/[^\n]*|\/\*[\s\S]*?\*\//g;
	for (const m of src.matchAll(re)) out += m[1] ?? m[2] ?? m[3] ?? '';
	return out;
};

const argvList = (name: string): string[] => {
	const out: string[] = [];
	const a = process.argv.slice(2);
	for (let i = 0; i < a.length; i++) if (a[i] === `--${name}` && a[i + 1] !== undefined) out.push(a[++i]);
	return out;
};
const argOne = (name: string): string | undefined => argvList(name).at(-1);

await runCli(async () => {
	const font = argOne('font');
	const out = argOne('out');
	if (!font || !out) throw new UsageError('usage: bun tools/subset-font.ts --font <src> --out <dst.woff2> [--text-file f]... [--text "…"] [--no-ascii]');
	let text = argvList('text').join('');
	for (const f of argvList('text-file')) {
		const src = readFileSync(resolve(f), 'utf8');
		text += ['.ts', '.tsx', '.js', '.json'].includes(extname(f)) ? literals(src) : src;
	}
	if (!process.argv.includes('--no-ascii')) for (let c = 0x20; c < 0x7f; c++) text += String.fromCharCode(c);
	const chars = [...new Set(Array.from(text))].filter((c) => c !== '\n' && c !== '\r' && c !== '\t').sort();
	const tmp = mkdtempSync(join(tmpdir(), 'paper2d-subset-'));
	try {
		const textFile = join(tmp, 'text.txt');
		writeFileSync(textFile, chars.join(''));
		const tool = process.env.PYFTSUBSET ?? Bun.which('pyftsubset');
		const head = tool ? [tool] : Bun.which('uvx') ? ['uvx', '--from', 'fonttools[woff]', 'pyftsubset'] : null;
		if (!head) throw new Error('pyftsubset not found: pip install "fonttools[woff]", or install uv, or set PYFTSUBSET');
		const cmd = [
			...head, resolve(font),
			`--text-file=${textFile}`,
			'--flavor=woff2',
			`--output-file=${resolve(out)}`,
			'--layout-features=*',
			// keep every name record: copyright (0), licence (13), licence URL (14) travel with the subset
			'--name-IDs=*', '--name-legacy', '--name-languages=*',
			'--no-hinting',
		];
		const p = Bun.spawn(cmd, {stdout: 'inherit', stderr: 'inherit'});
		if ((await p.exited) !== 0) throw new Error(`pyftsubset failed (${p.exitCode})`);
	} finally {
		rmSync(tmp, {recursive: true, force: true});
	}
	const nonAscii = chars.filter((c) => c.charCodeAt(0) > 0x7e);
	console.log(`${resolve(out)}: ${chars.length} characters (${nonAscii.length} beyond ASCII: ${nonAscii.join('')}), ${(statSync(resolve(out)).size / 1024).toFixed(1)} KB`);
});

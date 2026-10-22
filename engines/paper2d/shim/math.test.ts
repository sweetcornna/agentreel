// bun test：垫片数值部分的自检。spring 对照一个独立的数值积分（RK4，小步长），measureSpring 按定义逐帧核对，
// random 核对确定性、取值范围与分布，interpolate / bezier 核对边界行为。
import {describe, expect, test} from 'bun:test';
import {Easing, interpolate, measureSpring, random, spring, type SpringConfig} from './math';

const FPS = 30;

/** m·x'' = −k·(x − 1) − d·x'，x(0) = 0、x'(0) = 0，用 RK4 积到 t 秒（与 math.ts 的解析解互不相干） */
const integrate = (c: {damping: number; mass: number; stiffness: number}, t: number, dt = 1e-5): number => {
	let x = 0;
	let v = 0;
	const acc = (x_: number, v_: number) => (-c.stiffness * (x_ - 1) - c.damping * v_) / c.mass;
	let left = t;
	while (left > 1e-15) {
		const h = Math.min(dt, left);
		const k1x = v, k1v = acc(x, v);
		const k2x = v + (h / 2) * k1v, k2v = acc(x + (h / 2) * k1x, v + (h / 2) * k1v);
		const k3x = v + (h / 2) * k2v, k3v = acc(x + (h / 2) * k2x, v + (h / 2) * k2v);
		const k4x = v + h * k3v, k4v = acc(x + h * k3x, v + h * k3v);
		x += (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
		v += (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
		left -= h;
	}
	return x;
};

const REGIMES: [string, Partial<SpringConfig>][] = [
	['欠阻尼（默认 d=10 k=100 m=1，ζ=0.5）', {}],
	['欠阻尼、弹得多（d=3）', {damping: 3}],
	['临界阻尼（d=20，ζ=1）', {damping: 20}],
	['过阻尼（d=60，ζ=3）', {damping: 60}],
	['质量与刚度都换（m=2.5 k=180 d=12）', {mass: 2.5, stiffness: 180, damping: 12}],
];

describe('spring：解析解与数值积分一致', () => {
	for (const [name, config] of REGIMES) {
		test(name, () => {
			const c = {damping: 10, mass: 1, stiffness: 100, ...config};
			for (const frame of [0, 1, 2.5, 7, 12.25, 20, 33, 47.6, 60, 90]) {
				expect(Math.abs(spring({frame, fps: FPS, config}) - integrate(c, frame / FPS))).toBeLessThan(1e-9);
			}
		});
	}
	test('第 0 帧正好是 from，负帧也是', () => {
		expect(spring({frame: 0, fps: FPS})).toBe(0);
		expect(spring({frame: -5, fps: FPS, from: 3, to: 9})).toBe(3);
	});
	test('from / to 线性映射', () => {
		for (const frame of [3, 10, 25]) {
			const p = spring({frame, fps: FPS});
			expect(spring({frame, fps: FPS, from: 200, to: -100})).toBeCloseTo(200 + (-100 - 200) * p, 12);
		}
	});
	test('参数检查', () => {
		expect(() => spring({frame: 1, fps: FPS, config: {damping: 0}})).toThrow();
		expect(() => spring({frame: 1, fps: 0})).toThrow();
		expect(() => spring({frame: Number.NaN, fps: FPS})).toThrow();
		expect(() => spring({frame: 1, fps: FPS, durationInFrames: 0})).toThrow();
	});
});

describe('spring：overshootClamping', () => {
	test('欠阻尼弹簧不越过 to，没越过的帧与不夹时相同', () => {
		let reached = false;
		for (let f = 0; f <= 120; f += 0.5) {
			const free = spring({frame: f, fps: FPS, config: {damping: 4}, from: 0, to: 100});
			const held = spring({frame: f, fps: FPS, config: {damping: 4, overshootClamping: true}, from: 0, to: 100});
			expect(held).toBeLessThanOrEqual(100);
			if (free <= 100) expect(held).toBe(free);
			else {
				reached = true;
				expect(held).toBe(100);
			}
		}
		expect(reached).toBe(true);
	});
	test('from > to 时同样不越过（往下走）', () => {
		for (let f = 0; f <= 120; f++) expect(spring({frame: f, fps: FPS, config: {damping: 4, overshootClamping: true}, from: 50, to: 10})).toBeGreaterThanOrEqual(10);
	});
});

describe('spring：delay、durationInFrames、reverse', () => {
	test('delay 只是平移，延迟期间停在 from', () => {
		for (const f of [0, 4, 9.5]) expect(spring({frame: f, fps: FPS, delay: 10, from: 5, to: 6})).toBe(5);
		for (const f of [10, 13, 22.5, 40]) expect(spring({frame: f, fps: FPS, delay: 10})).toBe(spring({frame: f - 10, fps: FPS}));
	});
	test('durationInFrames 把弹簧自己的落定时间拉成给定帧数，到点后正好是 to', () => {
		const natural = measureSpring({fps: FPS});
		for (const f of [1, 5, 17.5, 29]) {
			expect(spring({frame: f, fps: FPS, durationInFrames: 30})).toBeCloseTo(spring({frame: (f * natural) / 30, fps: FPS}), 12);
		}
		expect(spring({frame: 30, fps: FPS, durationInFrames: 30, from: 2, to: 7})).toBe(7);
		expect(spring({frame: 500, fps: FPS, durationInFrames: 30, from: 2, to: 7})).toBe(7);
	});
	test('reverse 是同一段动画倒着放（不带 durationInFrames 时长度是 measureSpring）', () => {
		const D = measureSpring({fps: FPS});
		for (const f of [0, 3, 11.5, D - 1, D, D + 10]) {
			expect(spring({frame: f, fps: FPS, reverse: true})).toBe(spring({frame: D - f, fps: FPS}));
		}
	});
	test('reverse + durationInFrames + delay：延迟期间是 to，结束时正好是 from', () => {
		const o = {fps: FPS, durationInFrames: 40, delay: 6, reverse: true, from: 10, to: 30};
		expect(spring({...o, frame: 0})).toBe(30);
		expect(spring({...o, frame: 6})).toBe(30);
		expect(spring({...o, frame: 46})).toBe(10);
		expect(spring({...o, frame: 80})).toBe(10);
		// 中途在动（默认弹簧会越过 to，倒放时也一样）
		const mid = spring({...o, frame: 26});
		expect(mid).not.toBe(10);
		expect(mid).not.toBe(30);
		// 倒放：第 6 + s 帧的值 = 正放第 40 − s 帧的值
		for (const s of [1, 9, 20, 33]) expect(spring({...o, frame: 6 + s})).toBe(spring({...o, reverse: false, delay: 0, frame: 40 - s}));
	});
});

describe('measureSpring', () => {
	const cases: [string, Partial<SpringConfig>, number][] = [
		['默认', {}, 0.005],
		['弹得多', {damping: 3}, 0.005],
		['临界', {damping: 20}, 0.005],
		['过阻尼', {damping: 60}, 0.005],
		['阈值放宽', {damping: 8}, 0.05],
		['重而软', {mass: 3, stiffness: 40, damping: 6}, 0.001],
	];
	for (const [name, config, threshold] of cases) {
		test(`${name}：返回的帧起一直在阈值内，前一帧不在`, () => {
			const n = measureSpring({fps: FPS, config, threshold});
			expect(n).toBeGreaterThan(0);
			expect(Math.abs(spring({frame: n - 1, fps: FPS, config}) - 1)).toBeGreaterThanOrEqual(threshold);
			for (let f = n; f < n + 30 * FPS; f++) expect(Math.abs(spring({frame: f, fps: FPS, config}) - 1)).toBeLessThan(threshold);
		});
	}
	test('阻尼越大（到临界为止）落定越快；帧率翻倍帧数约翻倍', () => {
		expect(measureSpring({fps: FPS, config: {damping: 5}})).toBeGreaterThan(measureSpring({fps: FPS, config: {damping: 15}}));
		const a = measureSpring({fps: 30});
		const b = measureSpring({fps: 60});
		expect(Math.abs(b - 2 * a)).toBeLessThanOrEqual(2);
	});
	test('参数检查', () => {
		expect(() => measureSpring({fps: FPS, threshold: 0})).toThrow();
		expect(() => measureSpring({fps: FPS, config: {damping: -1}})).toThrow();
		expect(() => measureSpring({fps: -30})).toThrow();
	});
});

describe('random', () => {
	test('同一个种子总是同一个数（固定值，防止无意中换了序列）', () => {
		const golden: [string | number, number][] = [
			['a', random('a')],
			['gx0', random('gx0')],
			[42, random(42)],
		];
		for (const [seed, v] of golden) for (let i = 0; i < 3; i++) expect(random(seed)).toBe(v);
		expect(random('gx0')).toBe(GOLDEN.gx0);
		expect(random('ink-217.60000000000002')).toBe(GOLDEN.ink);
		expect(random(42)).toBe(GOLDEN.n42);
		expect(random(0.5)).toBe(GOLDEN.half);
	});
	test('取值在 [0, 1)，不同种子不同值，字符串与数字分开', () => {
		const seen = new Set<number>();
		for (let i = 0; i < 2000; i++) {
			const v = random(`gx${i}`);
			expect(v).toBeGreaterThanOrEqual(0);
			expect(v).toBeLessThan(1);
			seen.add(v);
		}
		expect(seen.size).toBe(2000);
		expect(random('1')).not.toBe(random(1));
		expect(random(217.6)).not.toBe(random(217.60000000000002));
		expect(random(0)).toBe(random(-0));
	});
	test('分布大致均匀（2 万个相邻种子，10 个桶）', () => {
		const n = 20000;
		const buckets = new Array(10).fill(0);
		let sum = 0;
		for (let i = 0; i < n; i++) {
			const v = random(`gy${i}`);
			sum += v;
			buckets[Math.floor(v * 10)]++;
		}
		expect(Math.abs(sum / n - 0.5)).toBeLessThan(0.01);
		const chi = buckets.reduce((a, b) => a + (b - n / 10) ** 2 / (n / 10), 0);
		expect(chi).toBeLessThan(30); // 9 个自由度，p ≈ 0.0004
	});
	test('null 给真随机，NaN 和别的类型报错', () => {
		const v = random(null);
		expect(v).toBeGreaterThanOrEqual(0);
		expect(v).toBeLessThan(1);
		expect(() => random(Number.NaN)).toThrow();
		expect(() => random({} as unknown as string)).toThrow();
	});
});

describe('interpolate 与 Easing', () => {
	test('分段、外推', () => {
		expect(interpolate(5, [0, 10], [0, 100])).toBe(50);
		expect(interpolate(15, [0, 10], [0, 100])).toBe(150);
		expect(interpolate(15, [0, 10], [0, 100], {extrapolateRight: 'clamp'})).toBe(100);
		expect(interpolate(-3, [0, 10], [0, 100], {extrapolateLeft: 'identity'})).toBe(-3);
		expect(interpolate(13, [0, 10], [0, 100], {extrapolateRight: 'wrap'})).toBeCloseTo(30, 12);
		expect(interpolate(-2, [0, 10], [0, 100], {extrapolateLeft: 'wrap'})).toBeCloseTo(80, 12);
		expect(interpolate(25, [0, 10, 20], [0, 100, 0], {extrapolateRight: 'wrap'})).toBeCloseTo(50, 12);
		expect(interpolate(15, [0, 10, 20], [0, 100, 0])).toBe(50);
	});
	test('节点属于左边那段（缓动在 0 处不为 0 时也一样）', () => {
		expect(interpolate(10, [0, 10, 20], [0, 1, 2], {easing: Easing.exp})).toBe(1);
	});
	test('参数检查', () => {
		expect(() => interpolate(1, [0, 0], [0, 1])).toThrow();
		expect(() => interpolate(1, [0, 1], [0, 1, 2])).toThrow();
		expect(() => interpolate(1, [0, Number.POSITIVE_INFINITY], [0, 1])).toThrow();
	});
	test('bezier：端点精确、单调、对称曲线中点 0.5', () => {
		const e = Easing.bezier(0.42, 0, 0.58, 1);
		expect(e(0)).toBe(0);
		expect(e(1)).toBe(1);
		expect(e(0.5)).toBeCloseTo(0.5, 6);
		let prev = 0;
		for (let t = 0; t <= 1; t += 0.01) {
			const v = e(t);
			expect(v).toBeGreaterThanOrEqual(prev - 1e-12);
			prev = v;
		}
		expect(() => Easing.bezier(1.2, 0, 0.5, 1)).toThrow();
	});
});

/** 当前实现的几个固定输出（改了哈希或生成器就会变：那样所有按名字取的随机量都会换，示例片要重渲） */
const GOLDEN = {gx0: 0.4615362964104861, ink: 0.7922054934315383, n42: 0.8664194268640131, half: 0.8630987028591335};

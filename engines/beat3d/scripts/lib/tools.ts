// 脚本要调用的外部工具：Google Chrome（或 Chromium）、ffmpeg、ffprobe。不写死在某一台机器上：
// 先看环境变量，再看这个平台的常见安装位置，最后在 PATH 里找。
//   CHROME_PATH        Chrome / Chromium 可执行文件
//   FFMPEG, FFPROBE    ffmpeg / ffprobe 可执行文件（或 PATH 里的名字）
//   BEAT3D_GPU         无头 Chrome 的 GPU 后端预设：metal | d3d11 | vulkan | gl | swiftshader
//                      （按平台默认：macOS metal，Windows d3d11，Linux vulkan）
//   BEAT3D_GPU_FLAGS   直接给 GPU 相关的命令行参数，空格分隔（优先于 BEAT3D_GPU）

import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const isWin = process.platform === 'win32';

/** `candidates` 里第一个存在的路径；都没有就在 PATH 里找 `names`；再没有就是 null。 */
function locate(candidates: string[], names: string[]): string | null {
  for (const c of candidates) if (c && existsSync(c)) return c;
  return onPath(names);
}

/** PATH 里第一个找得到的 `names`，没有就是 null。 */
function onPath(names: string[]): string | null {
  for (const n of names) {
    const hit = Bun.which(n);
    if (hit) return hit;
  }
  return null;
}

function required(what: string, env: string, path: string | null, hint: string): string {
  if (path) return path;
  throw new Error(`${what} not found. Set ${env} to its executable (${hint}).`);
}

let chrome: string | null | undefined;
/** Chrome / Chromium 可执行文件。 */
export function chromePath(): string {
  if (chrome === undefined) {
    const env = process.env.CHROME_PATH;
    const pf = process.env.PROGRAMFILES ?? 'C:\\Program Files';
    const pf86 = process.env['PROGRAMFILES(X86)'] ?? 'C:\\Program Files (x86)';
    const local = process.env.LOCALAPPDATA ?? join(homedir(), 'AppData', 'Local');
    const candidates =
      process.platform === 'darwin'
        ? [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            join(homedir(), 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
            '/Applications/Chromium.app/Contents/MacOS/Chromium',
            '/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
          ]
        : isWin
          ? [join(pf, 'Google\\Chrome\\Application\\chrome.exe'), join(pf86, 'Google\\Chrome\\Application\\chrome.exe'), join(local, 'Google\\Chrome\\Application\\chrome.exe')]
          : ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/opt/google/chrome/chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/snap/bin/chromium'];
    chrome = env ? (existsSync(env) ? env : null) : locate(candidates, ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome']);
    if (env && !chrome) throw new Error(`CHROME_PATH=${env} does not exist`);
  }
  return required('Google Chrome', 'CHROME_PATH', chrome, 'Chrome 120+ or Chromium with WebGL2');
}

const tool = (env: string, name: string): (() => string) => {
  let cached: string | null | undefined;
  return () => {
    if (cached === undefined) {
      const v = process.env[env];
      if (v) {
        if (!existsSync(v) && !Bun.which(v)) throw new Error(`${env}=${v} does not exist`);
        cached = existsSync(v) ? v : Bun.which(v);
      } else {
        // 先 PATH；PATH 里没有（例如从图形界面启动、PATH 不全）再看常见的安装位置（Homebrew、/usr/local、/usr）。
        cached = onPath([name]) ?? locate(isWin ? [] : [`/opt/homebrew/bin/${name}`, `/usr/local/bin/${name}`, `/usr/bin/${name}`], []);
      }
    }
    return required(name, env, cached ?? null, `${name} 6+ with libx264`);
  };
};
/** ffmpeg（要带 libx264）与 ffprobe。 */
export const ffmpeg = tool('FFMPEG', 'ffmpeg');
export const ffprobe = tool('FFPROBE', 'ffprobe');

const GPU_PRESETS: Record<string, string[]> = {
  // Apple 芯片上实测：--headless=new 下经 ANGLE 的 Metal 后端在 GPU 上有 WebGL2 + EXT_color_buffer_float
  // （那里 --use-angle=gl 拿不到 WebGL2）。其余预设按各平台 ANGLE 的常规后端给，未逐一实测。
  metal: ['--use-angle=metal'],
  d3d11: ['--use-angle=d3d11'],
  vulkan: ['--use-angle=vulkan', '--enable-features=Vulkan'],
  gl: ['--use-angle=gl'],
  // CPU 光栅化：哪里都能跑（CI、没有 GPU 的服务器），慢很多倍。
  swiftshader: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
};

/** 当前用的 GPU 预设名。 */
export function gpuPreset(): string {
  if (process.env.BEAT3D_GPU_FLAGS) return 'custom';
  const env = process.env.BEAT3D_GPU;
  if (env) {
    if (!GPU_PRESETS[env]) throw new Error(`BEAT3D_GPU=${env}: expected one of ${Object.keys(GPU_PRESETS).join(', ')}`);
    return env;
  }
  return process.platform === 'darwin' ? 'metal' : isWin ? 'd3d11' : 'vulkan';
}

/** 无头 Chrome 的 GPU 参数。 */
export function gpuFlags(): string[] {
  const custom = process.env.BEAT3D_GPU_FLAGS;
  const backend = custom ? custom.split(/\s+/).filter(Boolean) : GPU_PRESETS[gpuPreset()];
  return ['--headless=new', ...backend, '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-driver-bug-workarounds'];
}

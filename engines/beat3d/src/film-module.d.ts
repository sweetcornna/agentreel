// 正在构建的片子：scripts/build.ts 把 `@film` 解析到 films/<name>/scenes/index.ts（由 --film 选）。
declare module '@film' {
  import type { FilmDef } from '@beat3d/core/types';
  const film: FilmDef;
  export default film;
}

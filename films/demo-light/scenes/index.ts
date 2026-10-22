// 示例片《光与弦》：beat3d 引擎读这个文件的默认导出（FilmDef）。
// 一个程序（take.ts）一镜到底拍完 S01–S04；片子的胶片质感在这里给，镜头程序每帧再叠自己的。

import type { FilmDef } from '@beat3d/core/types';
import { take } from './take';
import { GRADE } from './palette';

const film: FilmDef = {
  programs: [take],
  // 默认光圈：圆形。S01 的光点自己按帧换光圈（叶形张开成圆形，lights.ts 的 apertureAt）。
  aperture: { shape: 'circle' },
  post: {
    // 这部片不让蓝色进画面：色差只往外溢暖色，光晕与横向耀斑都偏琥珀。
    chromaticStyle: 'warm',
    halation: { amount: 0.2, threshold: 0.9, tint: [1.0, 0.42, 0.12] },
    streak: { amount: 0.1, threshold: 3.0, falloff: 0.84, tint: [1.0, 0.6, 0.32] },
    bloom: { mix: 0.05, tint: [1.0, 0.94, 0.86] },
    grain: { amount: 0.012, size: 0.85, chroma: 0.05 },
    vignette: { amount: 0.26, softness: 0.6 },
    // AgX 的默认 look 会把琥珀压成米色：在编码空间里多给一点饱和度。
    grade: GRADE,
  },
  shotPost: {
    // 走廊与片尾的横向耀斑长一点（弦与光斑是全片最亮的点）。
    S03: { streak: { falloff: 0.88 } },
    S04: { streak: { falloff: 0.88 } },
  },
};

export default film;

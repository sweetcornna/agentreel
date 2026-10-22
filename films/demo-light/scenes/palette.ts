// 示例片的颜色：墨色的空间、一种暖光（琥珀）和它烧到最亮时的暖白。都是 Rec.709 线性值。
// INK 是「显示出来正好是 #0D0C10」的场景线性值（引擎 lib/color 的 sceneFor 反解显示变换），
// 暗处的雾按它校准（world/haze.ts 的 ambientFor），调色路径（post/gradePath.ts）也以它为起点。

import { lin, sceneFor, type RGB } from '@beat3d/lib/color';

/** 片子的显示调色（index.ts 的 post.grade 用它；sceneFor 反解时也要用同一份）。 */
export const GRADE = { lookSaturation: 1.25 };

export const PAL = {
  /** 琥珀：点光源、光丝、字的本色 */
  amber: lin('#FFA24E'),
  /** 灯：光斑与汇聚点（琥珀与深琥珀之间，AgX 下仍读成琥珀而不是米色） */
  lamp: lin('#F38A3A'),
  /** 深琥珀：光丝远端、暗处的光 */
  deep: lin('#E06A26'),
  /** 最亮的暖光（很亮的琥珀烧成的颜色） */
  warmWhite: [1.0, 0.78, 0.56] as RGB,
};

/** 显示出来正好是设计色的场景线性值。 */
export const SHOWS = {
  /** 空间的墨色 #0D0C10 */
  ink: sceneFor('#0D0C10', GRADE),
};

/** 暗光的色相：比琥珀更黄一点，经 AgX 趾部显示时才不发红（调色路径与雾里的光都用它）。 */
export const DIM_HUE: RGB = [1, 0.5, 0.16];

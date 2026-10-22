// Placeholder program: black with a slate (shot id, Chinese name, bar range, live position).
// Used for every shot that no program claims yet, and for the whole film in skeleton mode.

import type { ProgramDef, ShotProgram } from '../core/types';
import { CLOCK } from '../lib/time';
import { FONT, fontCss } from '../text/fonts';

export const placeholderProgram: ProgramDef = {
  id: 'placeholder',
  shots: [],
  create(): ShotProgram {
    return {
      frame({ frame, shot, layout }) {
        const s = layout.scale;
        return {
          overlay: (g) => {
            g.save();
            g.fillStyle = 'rgba(200,196,188,0.85)';
            g.textAlign = 'left';
            g.textBaseline = 'alphabetic';
            const x = layout.width * 0.06;
            const y = layout.visibleTop + 64 * s;
            g.font = fontCss(FONT.mono, 400, 30 * s);
            g.fillText(shot.id, x, y);
            g.font = fontCss(FONT.sansSC, 400, 30 * s);
            g.fillText(shot.zh, x + 110 * s, y);
            g.font = fontCss(FONT.mono, 300, 17 * s);
            g.fillStyle = 'rgba(200,196,188,0.6)';
            g.fillText(`${shot.from} – ${shot.to}   frames ${shot.fromFrame}–${shot.toFrame - 1}   ${shot.lens ?? shot.owner ?? ''}`, x, y + 34 * s);
            g.fillText(`now ${CLOCK.label(frame)}   f${frame}`, x, y + 60 * s);
            g.font = fontCss(FONT.mono, 300, 14 * s);
            g.fillStyle = 'rgba(200,196,188,0.4)';
            g.fillText((shot.content ?? '').slice(0, 150), x, layout.visibleBottom - 36 * s);
            g.restore();
          },
        };
      },
      dispose() {},
    };
  },
};

"""重建 sampler.birds（以及 nature.sfx('chirp')）用的鸟鸣单句目录。

    python music/sampler/tools/catalog_birds.py

分析采样根 field/ 里的四条鸟鸣录音（方法与「clear」的标准见 sampler/birds.py），写 <采样根>/field/bird_calls.json。
确定性，几秒钟。fetch_samples.py 取 field 包时会自动跑一次；录音本身不改。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampler import birds  # noqa: E402

if __name__ == '__main__':
    rows = birds.write_catalog()
    clear = [r for r in rows if r['clear']]
    print(f'{len(rows)} calls, {len(clear)} clear -> {birds.CATALOG}')

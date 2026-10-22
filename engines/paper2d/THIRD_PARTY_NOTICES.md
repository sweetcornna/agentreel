# Third-party notices · 第三方署名

`engines/paper2d` 是 MIT 许可（仓库根目录 `LICENSE`）。下面这些部分来自别的项目或采用了公开的算法，按各自的许可在这里署名。

## React Native（MIT）

`shim/math.ts` 里的三次贝塞尔求解器（`bezier`）与缓动函数（`Easing`）跟随 React Native 的 Animated 库：
[`packages/react-native/Libraries/Animated/bezier.js`](https://github.com/facebook/react-native/blob/main/packages/react-native/Libraries/Animated/bezier.js) 与
[`Easing.js`](https://github.com/facebook/react-native/blob/main/packages/react-native/Libraries/Animated/Easing.js)，
常数、步骤与公式相同（改写成 TypeScript，并改了名字和错误信息）。`interpolate` 的外推语义（extend / identity / clamp、
节点归左边那段）与 React Native 的 Animated 插值相同，代码是另写的。许可原文：

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## bezier-easing（MIT）

React Native 的 `bezier.js` 是 [bezier-easing](https://github.com/gre/bezier-easing) 的移植；`shim/math.ts` 的贝塞尔求解器因此也源自它。许可（MIT）原文：

```
Copyright (c) 2014 Gaëtan Renaudeau

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
```

## mulberry32（公有领域）

`shim/math.ts` 的 `random` 用 mulberry32 由种子生成 [0, 1) 的数。mulberry32 由 Tommy Ettinger 于 2017 年写成并贡献到公有领域。

## FNV-1a（公有领域）

`shim/math.ts` 的 `random` 用 32 位 FNV-1a 把种子（字符串的 UTF-16 字节，或数字的 8 个字节）散列成 mulberry32 的初始状态。
FNV 散列由 Glenn Fowler、Landon Curt Noll、Kiem-Phong Vo 提出，属于公有领域。

## 字体

示例片用到的字体（OFL-1.1）不在引擎里，署名与许可全文见 `films/demo-paper/public/fonts/OFL.txt`。

// 手绘组件库入口：片里 import {Stroke, HandCircle, ...} from 'paper2d/components'。
export {Stroke, useBoil} from './Stroke';
export type {StrokeProps} from './Stroke';
export {HandArrow, HandCircle, HandSlash, HandTick, HandUnderline} from './shapes';
export type {MarkProps} from './shapes';
export {centerline, fbm1, hash, intersect, noise1, pacePos, polyLength, resample, roughen} from './geom';
export type {Pace, Pt, Texture} from './geom';

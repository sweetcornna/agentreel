// 片的入口：只登记根组件。引擎（engines/paper2d）从这里打包，'remotion' 在打包时指到引擎的垫片。
import {registerRoot} from 'remotion';
import {Root} from './Root';

registerRoot(Root);

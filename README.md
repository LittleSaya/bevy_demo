# Bevy Demos

初学 rust 和 bevy 过程中编写的 demo

## demo001_ball_on_the_plane

平面上的一个球

## demo002_ball_on_the_terrain

地形上的一个球，地形 mesh 是通过 heightmap 生成的，使用了 rtin 算法降低了顶点数

代码基本上都是从 [clynamen/bevy_terrain](https://github.com/clynamen/bevy_terrain) 抄来的 ，移除了 ui 、 飞行视角、线框和自定义的 shader ，改成使用 PbrBundle ，适配了 0.10.1 版本的 bevy

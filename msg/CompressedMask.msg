int32[] data           # 压缩后的数据（状态切换位置）
float32[4] box         # 边界框坐标 [x1, y1, x2, y2]
string label          # mask 的标签
float32 score         # mask 的置信度分数
int32[2] shape        # mask 的形状 [height, width] 
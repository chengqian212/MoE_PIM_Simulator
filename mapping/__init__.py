"""
第四步：真实逻辑权重恢复与延迟感知映射。

前三步只进行匿名空间规划。

本 package 从第四步开始引入：

- layer_id
- expert_id
- matrix_name
- Shared Expert
- LogicalWeightCube

后续再负责：

- LogicalWeightCube -> LogicalPlane
- LogicalPlane -> Sub-Cube
- PhysicalSlot 绑定
"""
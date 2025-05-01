# WUST_VISION
* 2025年4月30日开始开发

## 已完成：
* 海康驱动直连观测器，已实现openvino和tensorrt基于相同模型的模板开发，165hz情况下openvino运算延迟在10ms-20ms实际帧率，165hz（对于nuc），120hz情况下tensorrt运算延迟在10-25ms，实际帧率100-110hz（对于6核心nx），位姿结算已完成。
* 科学的项目结构，一键编译部署脚本
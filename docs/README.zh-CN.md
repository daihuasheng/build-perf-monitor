# MyMonitor 文档

> **语言**: [English](README.md) | [中文](README.zh-CN.md)

欢迎查阅 MyMonitor 文档！本指南全面涵盖了使用和开发 MyMonitor 所需的所有信息。

## 快速入门

- **新用户**: 从[配置指南](configuration_guide.zh-CN.md)开始
- **开发者**: 查看[开发者指南](developer_guide.zh-CN.md)
- **故障排除**: 参考[故障排除指南](troubleshooting.zh-CN.md)

## 文档结构

### 用户指南

#### [配置指南](configuration_guide.zh-CN.md)
完整的 MyMonitor 项目配置指南，包括：
- 主要配置文件（`config.toml`、`projects.toml`、`rules.toml`）
- 存储设置和压缩选项
- CPU 调度和线程池配置
- 环境变量和最佳实践

#### [存储格式](storage_formats.zh-CN.md)
数据存储格式的综合指南：
- Parquet 与 CSV/JSON 性能对比
- 配置选项和压缩算法
- 数据转换工具和迁移策略
- 数据访问和分析的最佳实践

#### [故障排除指南](troubleshooting.zh-CN.md)
常见问题的解决方案：
- 安装和依赖问题
- 配置错误和验证问题
- 性能优化技术
- 特定系统的故障排除（macOS、Windows、容器）

### 开发者资源

#### [API 参考](api_reference.zh-CN.md)
完整的 API 文档，涵盖：
- 核心类（`DataStorageManager`、`AbstractMemoryCollector`、`ProcessClassifier`）
- 配置类和数据模型
- 实用函数和错误处理
- 线程安全和性能考虑

#### [开发者指南](developer_guide.zh-CN.md)
贡献者必备信息：
- 开发环境设置
- 编码标准和风格指南
- 测试实践和框架
- 添加新功能（存储后端、收集器）
- 性能考虑和调试技术

## 功能亮点

### 高性能存储
- **Parquet 格式**: 与 CSV/JSON 相比节省 75-80% 存储空间
- **快速查询**: 通过列裁剪实现 3-5 倍性能提升
- **多种压缩**: Snappy、Gzip、Brotli、LZ4、Zstd 选项
- **向后兼容**: 可选的传统格式生成

### 高级监控
- **实时收集**: 可配置间隔的 PSS/RSS 内存监控
- **进程分类**: 基于规则引擎的自动分类
- **CPU 亲和性**: 智能核心分配以获得一致的测量结果
- **线程池管理**: 优化的并行处理

### 开发者友好
- **模块化架构**: 关注点的清晰分离
- **全面测试**: 135+ 测试用例，完全覆盖
- **类型安全**: 整个代码库中完整的类型提示
- **可扩展设计**: 易于添加新的收集器和存储后端

## 获取帮助

### 文档问题
如果您发现文档中的错误或缺失信息：
1. 首先查看[故障排除指南](troubleshooting.zh-CN.md)
2. 搜索仓库中的现有问题
3. 创建带有 `documentation` 标签的新问题

### 技术支持
对于技术问题和支持：
1. 查看 [API 参考](api_reference.zh-CN.md)获取详细的函数文档
2. 检查[配置指南](configuration_guide.zh-CN.md)解决设置问题
3. 启用详细日志记录以获取调试信息

### 贡献
有兴趣贡献代码？请参阅[开发者指南](developer_guide.zh-CN.md)了解：
- 开发环境设置
- 编码标准和指南
- 测试要求
- 拉取请求流程

## 版本信息

本文档适用于 MyMonitor v2.1+，包括：
- Parquet 存储系统
- 增强的配置管理
- 改进的错误处理
- 全面的测试覆盖

对于旧版本，请参考相应的 git 标签。

## 许可证

MyMonitor 在 MIT 许可证下发布。详情请参阅 LICENSE 文件。

---

**快速导航:**
- [配置指南](configuration_guide.zh-CN.md) - 为您的项目设置 MyMonitor
- [存储格式](storage_formats.zh-CN.md) - 了解数据存储选项
- [API 参考](api_reference.zh-CN.md) - 完整的函数和类文档
- [开发者指南](developer_guide.zh-CN.md) - 为 MyMonitor 开发做贡献
- [故障排除](troubleshooting.zh-CN.md) - 解决常见问题

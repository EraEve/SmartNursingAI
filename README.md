# 🏥 智弈护晓 DawnGuard — 智医算

<div align="center">

**多模态感知 & 多智能体博弈驱动的医疗免陪照护智能护理机器人预测系统**

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![ACM Published](https://img.shields.io/badge/ACM-SHWID%202025-brightgreen)](https://dl.acm.org/doi/10.1145/3788112.3788145)
[![National Innovation](https://img.shields.io/badge/国家-大创项目-red)](#)
[![Challenge Cup](https://img.shields.io/badge/挑战杯-校级金奖-gold)](#)

</div>

## 项目简介

针对我国超 **4500 万** 失能老人的照护缺口，智弈护晓（智医算）项目依托自主研发的 **MAPFM（Multi-Agent Retrieval-Augmented Framework for Medical Text Classification）** 多智能体检索增强框架，打造非接触、高精度、零死角的数字化免陪病房方案。

项目由山东财经大学跨专业大二团队发起，已获 2026 年国家级大创立项、"挑战杯"校级金奖，ACM 论文成果见刊 SHWID 2025（EI/Scopus 检索，ISBN: 979-8-4007-1847-2），并已落地济南 5 家医院。

## 核心指标

| 指标 | 数值 |
|------|------|
| MAPFM 医疗文本分类精确率 | **81.95%** |
| 端侧预警响应延迟 | **< 0.85s** |
| 护工照护服务半径 | **1:15** |
| 单院年均节约成本 | **≈ 57 万元** |

## 技术架构

### MAPFM 多智能体框架

专为医疗场景小样本数据设计的核心技术底座：

- **DMA（Decision Medical Agent）** — 提取病历特征，负责初筛推理
- **RAA（Retrieval-Augmented Agent）** — 校验医学知识库上下文
- **Nash 均衡博弈协同** — 双智能体动态对抗交互，过滤翻身/咳嗽等生理噪声，降低虚警率
- **Adaptive-RAG 策略** — 向量化接入 PubMed、UpToDate 等权威知识库，确保循证基础

### 端侧部署

- 深度适配 Meditron-7B 与 Llama-3 医疗私有模型
- 4-bit 量化 + 模型剪枝 + 知识蒸馏
- 边缘服务器本地秒级推理
- 联邦学习保证患者隐私不出病房，符合等保三级

## 项目结构

```
SmartNursingAI/
├── index.html              # 单文件开源官网（Tailwind + ECharts）
├── assets/                 # 静态资源
├── MAPFM.png               # 技术架构图
├── SWOT.png                # SWOT 战略分析图
├── 4-6.png                 # 应用场景矩阵图
├── "智医算...项目计划书.pdf"  # 完整商业计划书
├── 混合检索.py              # 混合检索算法实现
├── 自适应.py               # 自适应 RAG 策略实现
├── 重排序.py               # 重排序算法实现
├── 混合检索.pdf             # 混合检索论文
├── 自适应.pdf               # 自适应 RAG 论文
├── 重排序.pdf               # 重排序论文
└── 代码                     # 核心算法代码
```

## 快速开始

### 本地预览官网

```bash
# 方法一：直接用浏览器打开
start index.html

# 方法二：使用任意静态服务器
python -m http.server 8080
# 浏览器访问 http://localhost:8080
```

### 二次开发

官网基于纯 HTML + Tailwind CSS (CDN) + ECharts (CDN) 构建，无需构建工具：

1. 直接编辑 `index.html`
2. Tailwind 样式通过 CDN 即时编译，修改 class 即可生效
3. 邮件表单：配置 `APP_CONFIG.emailjs` 对象启用真实邮件发送

## 已落地医院

| 医院/机构 | 类型 | 核心成效 |
|-----------|------|----------|
| 济南复大肿瘤医院 | 三甲·肿瘤特护 | 单院年节约 57 万，操作合规率 98.7% |
| 济南华圣医院 | 综合·老年评估 | 风险预警准确率 ≥90% |
| 秀水情养老中心 | 大型康养机构 | 基础耗时缩减 30%，压疮环比下降 22% |
| 潍坊经开卫生中心 | 社区卫生 | 文书处理耗时下降 80% |
| 沂城街道卫生中心 | 基层医疗 | 慢病数据动态采集，1:N 远程协同 |

## 团队成员

| 姓名 | 角色 | 专业 |
|------|------|------|
| 谷元杰 | 算法统筹 / CEO | 计算机科学与技术 |
| 卢子涵 | 临床对接 / 商务 | 信息管理与系统 |
| 张芸嘉 | 前端开发 / 可视化 | 计算机科学与技术 |
| 陈炎希 | 商业运营 / 市场 | 经济学科拔尖人才 |
| 李玥莹 | 财务精算 / CFO | 金融数学 |
| 于海桐 | 后端开发 / 数据库 | 应用数学 |
| 张涵 | 技术测试 / QA | 计算机科学与技术 |
| 赵梓含 | 项目管理 | 公共管理 |
| 祝萌璐 | 学术文献 / 科研 | 公共管理类 |
| 张宝瑄 | UI/UX 设计 / PM | 计算机科学与技术 |

**指导教师：** 刘峥 教授（博导）、李珊珊 讲师（双创导师）

## 学术成果与荣誉

- 🏅 **ACM SHWID 2025** 正式见刊（EI/Scopus 检索）— *MAPFM: A Multi-Agent Retrieval-Augmented Framework for Medical Text Classification*
- 🏅 **2026 国家级大学生创新创业项目** 立项
- 🏅 **"挑战杯"** 院级特等奖 / 校级金奖
- 🏅 国家发明专利 1 项（实质审查阶段）
- 🏅 软件著作权 3 项

## 商业模式

四阶多元协同盈利：

1. **硬件入驻** — 租赁与销售（月租 3.5 万/台，押金 20 万）
2. **SaaS 订阅** — 按床位月度收费（三甲 800 元/床，二级 600 元/床）
3. **数据变现** — 算法定制 20-50 万/次，年度报告 15 万/年
4. **技术授权** — API 接口 10-50 万/年，护理师培训认证 1800-6800 元/人

## 融资规划

| 阶段 | 金额 | 估值 | 用途 |
|------|------|------|------|
| 种子轮 | 1000 万 | 4000 万（出让 20%） | 原型机迭代、3 家示范病房打样 |
| A 轮 | 3000 万 | 1.2 亿（出让 20%） | 二类器械注册、覆盖 15+ 机构 |
| B 轮 | 8000 万 | 4 亿（出让 16.6%） | 全国 50+ 医院、冲刺 IPO |

## 许可证

本项目采用 [MIT License](LICENSE) 开源 —— 算法代码、官网、商业计划书均完全开放，欢迎二次开发、学术引用与产业落地。

## 联系方式

- 📧 官方邮箱：[yuanjiegu52@gmail.com](mailto:yuanjiegu52@gmail.com)
- 🌐 官网：[https://eraeve.github.io/zhiyisuan--SDUFE.github.io/](https://eraeve.github.io/zhiyisuan--SDUFE.github.io/)
- 📍 研发总部：山东省济南市舜耕路 40 号 山东财经大学双创中心
- 📄 [ACM 论文](https://dl.acm.org/doi/10.1145/3788112.3788145) | [GitHub 仓库](https://github.com/EraEve/SmartNursingAI)

---

<p align="center"><b>智护未来，仁济民生</b> — 让每一张病床都被 24 小时温柔凝视。</p>

# RQA2025 移动端应用

## 项目概述

RQA2025量化交易系统的移动端应用，支持iOS和Android平台。

## 技术栈

- **前端框架**: React Native 0.72+
- **状态管理**: Redux Toolkit + RTK Query
- **导航**: React Navigation 6
- **图表**: react-native-chart-kit
- **推送**: Firebase Cloud Messaging
- **生物识别**: react-native-biometrics
- **本地存储**: AsyncStorage

## 项目结构

```
mobile/
├── android/              # Android原生代码
├── ios/                  # iOS原生代码
├── src/
│   ├── api/             # API接口
│   ├── components/      # 组件
│   ├── screens/         # 页面
│   ├── store/           # Redux状态管理
│   ├── hooks/           # 自定义Hooks
│   ├── utils/           # 工具函数
│   └── constants/       # 常量
├── App.tsx              # 应用入口
└── package.json
```

## 功能特性

1. 实时行情查看
2. 交易信号推送
3. 投资组合管理
4. 生物识别登录
5. 智能提醒
6. 语音助手

## 开发环境

```bash
# 安装依赖
npm install

# 启动开发服务器
npm start

# 运行Android
npm run android

# 运行iOS
npm run ios
```

## 发布

```bash
# Android
npm run build:android

# iOS
npm run build:ios
```

/**
 * RQA 2.0 移动端导航类型定义
 *
 * 定义应用中使用的所有导航参数类型
 * 确保类型安全和更好的开发体验
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {NavigatorScreenParams} from '@react-navigation/native';

// 主栈导航参数
export type RootStackParamList = {
  Welcome: undefined;
  Login: undefined;
  Register: undefined;
  ForgotPassword: undefined;
  Main: NavigatorScreenParams<MainTabParamList>;
  StrategyDetail: {strategyId: string};
  TradeExecution: {symbol: string; action: 'buy' | 'sell'};
  Settings: undefined;
};

// 主标签导航参数
export type MainTabParamList = {
  Portfolio: undefined;
  Strategies: undefined;
  Trading: undefined;
  Analytics: undefined;
  Profile: undefined;
};

// 投资组合相关参数
export type PortfolioStackParamList = {
  PortfolioHome: undefined;
  AssetDetail: {assetId: string};
  PerformanceHistory: {assetId: string};
  Rebalancing: undefined;
};

// 策略相关参数
export type StrategiesStackParamList = {
  StrategiesHome: undefined;
  StrategyList: {category?: string};
  StrategyDetail: {strategyId: string};
  StrategyBacktest: {strategyId: string};
  StrategyComparison: {strategyIds: string[]};
};

// 交易相关参数
export type TradingStackParamList = {
  TradingHome: undefined;
  MarketData: undefined;
  OrderBook: {symbol: string};
  TradeHistory: undefined;
  Watchlist: undefined;
};

// 分析相关参数
export type AnalyticsStackParamList = {
  AnalyticsHome: undefined;
  PerformanceAnalysis: undefined;
  RiskAnalysis: undefined;
  MarketAnalysis: undefined;
  CustomReport: undefined;
};

// 个人资料相关参数
export type ProfileStackParamList = {
  ProfileHome: undefined;
  AccountSettings: undefined;
  SecuritySettings: undefined;
  NotificationSettings: undefined;
  SubscriptionManagement: undefined;
};





// RQA2025 Web平台类型定义

// 用户相关类型
export interface User {
  id: string
  email: string
  name: string
  avatar?: string
  role: UserRole
  permissions: Permission[]
  createdAt: Date
  updatedAt: Date
  isActive: boolean
  preferences: UserPreferences
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system'
  language: string
  timezone: string
  notifications: NotificationSettings
  dashboard: DashboardLayout
}

export interface NotificationSettings {
  email: boolean
  push: boolean
  sms: boolean
  trading: boolean
  system: boolean
}

export interface DashboardLayout {
  widgets: DashboardWidget[]
  layout: 'grid' | 'list'
  columns: number
}

export interface DashboardWidget {
  id: string
  type: WidgetType
  position: { x: number; y: number }
  size: { width: number; height: number }
  config: Record<string, any>
}

export type UserRole = 'admin' | 'trader' | 'analyst' | 'viewer'

export type Permission =
  | 'read:trading'
  | 'write:trading'
  | 'read:portfolio'
  | 'write:portfolio'
  | 'read:strategy'
  | 'write:strategy'
  | 'read:admin'
  | 'write:admin'

// 交易相关类型
export interface Trade {
  id: string
  symbol: string
  type: TradeType
  side: TradeSide
  quantity: number
  price: number
  executedAt: Date
  status: TradeStatus
  strategyId?: string
  userId: string
}

export type TradeType = 'market' | 'limit' | 'stop' | 'stop_limit'
export type TradeSide = 'buy' | 'sell'
export type TradeStatus = 'pending' | 'executed' | 'cancelled' | 'failed'

// 策略相关类型
export interface Strategy {
  id: string
  name: string
  description: string
  type: StrategyType
  config: StrategyConfig
  status: StrategyStatus
  performance: StrategyPerformance
  userId: string
  createdAt: Date
  updatedAt: Date
}

export type StrategyType =
  | 'trend_following'
  | 'mean_reversion'
  | 'momentum'
  | 'arbitrage'
  | 'ml_based'
  | 'rl_based'

export interface StrategyConfig {
  parameters: Record<string, any>
  indicators: IndicatorConfig[]
  rules: StrategyRule[]
  riskManagement: RiskConfig
}

export interface IndicatorConfig {
  name: string
  type: string
  parameters: Record<string, any>
  timeframe: string
}

export interface StrategyRule {
  id: string
  condition: string
  action: string
  priority: number
}

export interface RiskConfig {
  maxPosition: number
  maxDrawdown: number
  stopLoss: number
  takeProfit: number
}

export type StrategyStatus = 'active' | 'inactive' | 'backtesting' | 'error'

export interface StrategyPerformance {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  totalTrades: number
  avgTrade: number
}

// 投资组合相关类型
export interface Portfolio {
  id: string
  name: string
  userId: string
  positions: Position[]
  cash: number
  totalValue: number
  dayChange: number
  dayChangePercent: number
  createdAt: Date
  updatedAt: Date
}

export interface Position {
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  marketValue: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  dayChange: number
  dayChangePercent: number
}

// 市场数据类型
export interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  marketCap?: number
  pe?: number
  pb?: number
  dividend?: number
  timestamp: Date
}

export interface OHLCV {
  timestamp: Date
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// 图表组件类型
export interface ChartConfig {
  symbol: string
  timeframe: Timeframe
  indicators: ChartIndicator[]
  overlays: ChartOverlay[]
  studies: ChartStudy[]
}

export type Timeframe =
  | '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M'

export interface ChartIndicator {
  type: string
  name: string
  parameters: Record<string, any>
  color: string
  style: 'line' | 'area' | 'histogram'
}

export interface ChartOverlay {
  type: string
  data: any[]
  color: string
}

export interface ChartStudy {
  type: string
  parameters: Record<string, any>
}

// 仪表板组件类型
export type WidgetType =
  | 'portfolio_overview'
  | 'performance_chart'
  | 'risk_metrics'
  | 'recent_trades'
  | 'market_news'
  | 'strategy_monitor'
  | 'alerts'
  | 'watchlist'

// API响应类型
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
}

// WebSocket消息类型
export interface WSMessage {
  type: WSMessageType
  payload: any
  timestamp: Date
}

export type WSMessageType =
  | 'market_data'
  | 'trade_update'
  | 'portfolio_update'
  | 'strategy_signal'
  | 'system_alert'
  | 'user_notification'

// 表单类型
export interface LoginForm {
  email: string
  password: string
  rememberMe?: boolean
}

export interface RegisterForm {
  name: string
  email: string
  password: string
  confirmPassword: string
  agreeToTerms: boolean
}

export interface StrategyForm {
  name: string
  description: string
  type: StrategyType
  config: StrategyConfig
}

// 工具类型
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>


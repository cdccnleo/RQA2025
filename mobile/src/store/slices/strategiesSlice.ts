/**
 * RQA 2.0 策略状态管理
 *
 * 处理量化策略的获取、筛选、详情查看等功能
 * 支持策略性能分析和用户偏好设置
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';

// 类型定义
interface Strategy {
  id: string;
  name: string;
  description: string;
  category: 'momentum' | 'mean_reversion' | 'arbitrage' | 'ml_based' | 'risk_parity';
  riskLevel: 'low' | 'medium' | 'high';
  timeHorizon: 'short' | 'medium' | 'long';
  expectedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  backtestPeriod: string;
  isActive: boolean;
  popularity: number;
  author: string;
  createdAt: string;
  lastUpdated: string;
  tags: string[];
}

interface StrategyPerformance {
  returns: number[];
  dates: string[];
  benchmarkReturns: number[];
  drawdown: number[];
  monthlyReturns: number[];
}

interface StrategyFilter {
  category?: string;
  riskLevel?: string;
  timeHorizon?: string;
  minReturn?: number;
  maxVolatility?: number;
  sortBy?: 'popularity' | 'return' | 'sharpe' | 'winRate';
  sortOrder?: 'asc' | 'desc';
}

interface StrategiesState {
  strategies: Strategy[];
  filteredStrategies: Strategy[];
  selectedStrategy: Strategy | null;
  strategyPerformance: StrategyPerformance | null;
  filters: StrategyFilter;
  isLoading: boolean;
  error: string | null;
  currentPage: number;
  hasMore: boolean;
}

// 初始状态
const initialState: StrategiesState = {
  strategies: [],
  filteredStrategies: [],
  selectedStrategy: null,
  strategyPerformance: null,
  filters: {
    sortBy: 'popularity',
    sortOrder: 'desc',
  },
  isLoading: false,
  error: null,
  currentPage: 1,
  hasMore: true,
};

// 异步操作
export const fetchStrategies = createAsyncThunk(
  'strategies/fetchStrategies',
  async (params: {page?: number; limit?: number} = {}, {rejectWithValue}) => {
    try {
      const queryParams = new URLSearchParams({
        page: (params.page || 1).toString(),
        limit: (params.limit || 20).toString(),
      });

      const response = await fetch(`/api/strategies?${queryParams}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取策略列表失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const fetchStrategyDetails = createAsyncThunk(
  'strategies/fetchStrategyDetails',
  async (strategyId: string, {rejectWithValue}) => {
    try {
      const response = await fetch(`/api/strategies/${strategyId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取策略详情失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const fetchStrategyPerformance = createAsyncThunk(
  'strategies/fetchStrategyPerformance',
  async (strategyId: string, {rejectWithValue}) => {
    try {
      const response = await fetch(`/api/strategies/${strategyId}/performance`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取策略性能失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

// Strategies Slice
const strategiesSlice = createSlice({
  name: 'strategies',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null;
    },
    setSelectedStrategy: (state, action: PayloadAction<Strategy | null>) => {
      state.selectedStrategy = action.payload;
    },
    updateFilters: (state, action: PayloadAction<Partial<StrategyFilter>>) => {
      state.filters = {...state.filters, ...action.payload};
      state.filteredStrategies = applyFilters(state.strategies, state.filters);
    },
    clearFilters: state => {
      state.filters = {
        sortBy: 'popularity',
        sortOrder: 'desc',
      };
      state.filteredStrategies = state.strategies;
    },
    resetStrategies: state => {
      state.currentPage = 1;
      state.hasMore = true;
      state.strategies = [];
      state.filteredStrategies = [];
    },
  },
  extraReducers: builder => {
    // 获取策略列表
    builder
      .addCase(fetchStrategies.pending, state => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchStrategies.fulfilled, (state, action) => {
        state.isLoading = false;
        const newStrategies = action.payload.strategies;
        const page = action.meta.arg.page || 1;

        if (page === 1) {
          state.strategies = newStrategies;
        } else {
          state.strategies = [...state.strategies, ...newStrategies];
        }

        state.filteredStrategies = applyFilters(state.strategies, state.filters);
        state.currentPage = page;
        state.hasMore = newStrategies.length === (action.meta.arg.limit || 20);
      })
      .addCase(fetchStrategies.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // 获取策略详情
    builder
      .addCase(fetchStrategyDetails.fulfilled, (state, action) => {
        const strategy = action.payload;
        const index = state.strategies.findIndex(s => s.id === strategy.id);
        if (index !== -1) {
          state.strategies[index] = strategy;
        }
        state.selectedStrategy = strategy;
      });

    // 获取策略性能
    builder
      .addCase(fetchStrategyPerformance.fulfilled, (state, action) => {
        state.strategyPerformance = action.payload;
      });
  },
});

// 辅助函数：应用筛选和排序
function applyFilters(strategies: Strategy[], filters: StrategyFilter): Strategy[] {
  let filtered = [...strategies];

  // 应用筛选条件
  if (filters.category) {
    filtered = filtered.filter(s => s.category === filters.category);
  }

  if (filters.riskLevel) {
    filtered = filtered.filter(s => s.riskLevel === filters.riskLevel);
  }

  if (filters.timeHorizon) {
    filtered = filtered.filter(s => s.timeHorizon === filters.timeHorizon);
  }

  if (filters.minReturn !== undefined) {
    filtered = filtered.filter(s => s.expectedReturn >= filters.minReturn!);
  }

  if (filters.maxVolatility !== undefined) {
    filtered = filtered.filter(s => s.volatility <= filters.maxVolatility!);
  }

  // 应用排序
  const {sortBy = 'popularity', sortOrder = 'desc'} = filters;
  filtered.sort((a, b) => {
    let aValue: number, bValue: number;

    switch (sortBy) {
      case 'return':
        aValue = a.expectedReturn;
        bValue = b.expectedReturn;
        break;
      case 'sharpe':
        aValue = a.sharpeRatio;
        bValue = b.sharpeRatio;
        break;
      case 'winRate':
        aValue = a.winRate;
        bValue = b.winRate;
        break;
      case 'popularity':
      default:
        aValue = a.popularity;
        bValue = b.popularity;
        break;
    }

    return sortOrder === 'desc' ? bValue - aValue : aValue - bValue;
  });

  return filtered;
}

// 导出 actions
export const {
  clearError,
  setSelectedStrategy,
  updateFilters,
  clearFilters,
  resetStrategies,
} = strategiesSlice.actions;

// 导出 selectors
export const selectStrategies = (state: any) => state.strategies.filteredStrategies;
export const selectAllStrategies = (state: any) => state.strategies.strategies;
export const selectSelectedStrategy = (state: any) => state.strategies.selectedStrategy;
export const selectStrategyPerformance = (state: any) => state.strategies.strategyPerformance;
export const selectStrategiesLoading = (state: any) => state.strategies.isLoading;
export const selectStrategiesError = (state: any) => state.strategies.error;
export const selectStrategiesFilters = (state: any) => state.strategies.filters;
export const selectStrategiesHasMore = (state: any) => state.strategies.hasMore;

// 导出 reducer
export default strategiesSlice.reducer;





/**
 * RQA 2.0 投资组合状态管理
 *
 * 处理投资组合数据、资产管理、收益计算等核心功能
 * 支持实时数据更新和多资产类型管理
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';

// 类型定义
interface Asset {
  id: string;
  symbol: string;
  name: string;
  type: 'stock' | 'bond' | 'crypto' | 'fund' | 'option';
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  gainLoss: number;
  gainLossPercent: number;
  lastUpdated: string;
}

interface PortfolioSummary {
  totalValue: number;
  totalGainLoss: number;
  totalGainLossPercent: number;
  dayChange: number;
  dayChangePercent: number;
  cashBalance: number;
  buyingPower: number;
}

interface PortfolioState {
  assets: Asset[];
  summary: PortfolioSummary | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
  selectedAssetId: string | null;
}

// 初始状态
const initialState: PortfolioState = {
  assets: [],
  summary: null,
  isLoading: false,
  error: null,
  lastUpdated: null,
  selectedAssetId: null,
};

// 异步操作
export const fetchPortfolio = createAsyncThunk(
  'portfolio/fetchPortfolio',
  async (_, {rejectWithValue}) => {
    try {
      // 模拟API调用
      const response = await fetch('/api/portfolio', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取投资组合失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const fetchAssetDetails = createAsyncThunk(
  'portfolio/fetchAssetDetails',
  async (assetId: string, {rejectWithValue}) => {
    try {
      const response = await fetch(`/api/assets/${assetId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取资产详情失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const updateAssetPrices = createAsyncThunk(
  'portfolio/updateAssetPrices',
  async (_, {rejectWithValue}) => {
    try {
      const response = await fetch('/api/market/prices', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('更新价格失败');
      }

      const priceData = await response.json();
      return priceData;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

// Portfolio Slice
const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null;
    },
    setSelectedAsset: (state, action: PayloadAction<string | null>) => {
      state.selectedAssetId = action.payload;
    },
    updateAssetLocally: (state, action: PayloadAction<Asset>) => {
      const index = state.assets.findIndex(
        asset => asset.id === action.payload.id,
      );
      if (index !== -1) {
        state.assets[index] = action.payload;
        // 重新计算汇总数据
        state.summary = calculatePortfolioSummary(state.assets);
      }
    },
    refreshPortfolio: state => {
      state.lastUpdated = new Date().toISOString();
    },
  },
  extraReducers: builder => {
    // 获取投资组合
    builder
      .addCase(fetchPortfolio.pending, state => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchPortfolio.fulfilled, (state, action) => {
        state.isLoading = false;
        state.assets = action.payload.assets;
        state.summary = action.payload.summary;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchPortfolio.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // 获取资产详情
    builder
      .addCase(fetchAssetDetails.fulfilled, (state, action) => {
        const updatedAsset = action.payload;
        const index = state.assets.findIndex(
          asset => asset.id === updatedAsset.id,
        );
        if (index !== -1) {
          state.assets[index] = updatedAsset;
        }
      });

    // 更新资产价格
    builder
      .addCase(updateAssetPrices.fulfilled, (state, action) => {
        const priceUpdates = action.payload;
        state.assets = state.assets.map(asset => {
          const priceUpdate = priceUpdates[asset.symbol];
          if (priceUpdate) {
            const newPrice = priceUpdate.price;
            const gainLoss = (newPrice - asset.averagePrice) * asset.quantity;
            const gainLossPercent = ((newPrice - asset.averagePrice) / asset.averagePrice) * 100;

            return {
              ...asset,
              currentPrice: newPrice,
              marketValue: newPrice * asset.quantity,
              gainLoss,
              gainLossPercent,
              lastUpdated: new Date().toISOString(),
            };
          }
          return asset;
        });

        // 重新计算汇总数据
        state.summary = calculatePortfolioSummary(state.assets);
        state.lastUpdated = new Date().toISOString();
      });
  },
});

// 辅助函数：计算投资组合汇总
function calculatePortfolioSummary(assets: Asset[]): PortfolioSummary {
  const totalValue = assets.reduce((sum, asset) => sum + asset.marketValue, 0);
  const totalCost = assets.reduce((sum, asset) => sum + (asset.averagePrice * asset.quantity), 0);
  const totalGainLoss = totalValue - totalCost;
  const totalGainLossPercent = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0;

  // 模拟日变化 (实际项目中需要历史数据计算)
  const dayChange = totalGainLoss * 0.1; // 简化计算
  const dayChangePercent = totalValue > 0 ? (dayChange / totalValue) * 100 : 0;

  return {
    totalValue,
    totalGainLoss,
    totalGainLossPercent,
    dayChange,
    dayChangePercent,
    cashBalance: 10000, // 模拟数据
    buyingPower: 25000, // 模拟数据
  };
}

// 导出 actions
export const {
  clearError,
  setSelectedAsset,
  updateAssetLocally,
  refreshPortfolio,
} = portfolioSlice.actions;

// 导出 selectors
export const selectPortfolioAssets = (state: any) => state.portfolio.assets;
export const selectPortfolioSummary = (state: any) => state.portfolio.summary;
export const selectPortfolioLoading = (state: any) => state.portfolio.isLoading;
export const selectPortfolioError = (state: any) => state.portfolio.error;
export const selectSelectedAsset = (state: any) =>
  state.portfolio.assets.find((asset: Asset) => asset.id === state.portfolio.selectedAssetId);

// 导出 reducer
export default portfolioSlice.reducer;





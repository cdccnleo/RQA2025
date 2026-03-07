/**
 * RQA 2.0 交易状态管理
 *
 * 处理交易订单、执行状态、市场数据、交易历史等核心功能
 * 支持实时交易、订单管理和风险控制
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';

// 类型定义
interface Order {
  id: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  createdAt: string;
  updatedAt: string;
  filledQuantity: number;
  averageFillPrice?: number;
  commission: number;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
  timestamp: string;
}

interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  marketValue: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  realizedPnL: number;
}

interface TradingState {
  orders: Order[];
  positions: Position[];
  marketData: {[symbol: string]: MarketData};
  watchlist: string[];
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
  selectedSymbol: string | null;
}

// 初始状态
const initialState: TradingState = {
  orders: [],
  positions: [],
  marketData: {},
  watchlist: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
  isLoading: false,
  error: null,
  lastUpdated: null,
  selectedSymbol: null,
};

// 异步操作
export const fetchOrders = createAsyncThunk(
  'trading/fetchOrders',
  async (_, {rejectWithValue}) => {
    try {
      const response = await fetch('/api/trading/orders', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取订单失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const placeOrder = createAsyncThunk(
  'trading/placeOrder',
  async (orderData: {
    symbol: string;
    type: Order['type'];
    side: Order['side'];
    quantity: number;
    price?: number;
    stopPrice?: number;
  }, {rejectWithValue}) => {
    try {
      const response = await fetch('/api/trading/orders', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(orderData),
      });

      if (!response.ok) {
        throw new Error('下单失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const cancelOrder = createAsyncThunk(
  'trading/cancelOrder',
  async (orderId: string, {rejectWithValue}) => {
    try {
      const response = await fetch(`/api/trading/orders/${orderId}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('取消订单失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const fetchMarketData = createAsyncThunk(
  'trading/fetchMarketData',
  async (symbols: string[], {rejectWithValue}) => {
    try {
      const symbolsParam = symbols.join(',');
      const response = await fetch(`/api/market/data?symbols=${symbolsParam}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取市场数据失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const fetchPositions = createAsyncThunk(
  'trading/fetchPositions',
  async (_, {rejectWithValue}) => {
    try {
      const response = await fetch('/api/trading/positions', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('获取持仓失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

// Trading Slice
const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null;
    },
    setSelectedSymbol: (state, action: PayloadAction<string | null>) => {
      state.selectedSymbol = action.payload;
    },
    addToWatchlist: (state, action: PayloadAction<string>) => {
      if (!state.watchlist.includes(action.payload)) {
        state.watchlist.push(action.payload);
      }
    },
    removeFromWatchlist: (state, action: PayloadAction<string>) => {
      state.watchlist = state.watchlist.filter(symbol => symbol !== action.payload);
    },
    updateMarketData: (state, action: PayloadAction<{[symbol: string]: MarketData}>) => {
      state.marketData = {...state.marketData, ...action.payload};
      state.lastUpdated = new Date().toISOString();
    },
    clearOrders: state => {
      state.orders = [];
    },
  },
  extraReducers: builder => {
    // 获取订单
    builder
      .addCase(fetchOrders.pending, state => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchOrders.fulfilled, (state, action) => {
        state.isLoading = false;
        state.orders = action.payload.orders;
      })
      .addCase(fetchOrders.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // 下单
    builder
      .addCase(placeOrder.pending, state => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(placeOrder.fulfilled, (state, action) => {
        state.isLoading = false;
        state.orders.unshift(action.payload.order);
      })
      .addCase(placeOrder.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // 取消订单
    builder
      .addCase(cancelOrder.fulfilled, (state, action) => {
        const index = state.orders.findIndex(order => order.id === action.payload.orderId);
        if (index !== -1) {
          state.orders[index].status = 'cancelled';
          state.orders[index].updatedAt = new Date().toISOString();
        }
      });

    // 获取市场数据
    builder
      .addCase(fetchMarketData.fulfilled, (state, action) => {
        state.marketData = {...state.marketData, ...action.payload};
        state.lastUpdated = new Date().toISOString();
      });

    // 获取持仓
    builder
      .addCase(fetchPositions.fulfilled, (state, action) => {
        state.positions = action.payload.positions;
      });
  },
});

// 导出 actions
export const {
  clearError,
  setSelectedSymbol,
  addToWatchlist,
  removeFromWatchlist,
  updateMarketData,
  clearOrders,
} = tradingSlice.actions;

// 导出 selectors
export const selectOrders = (state: any) => state.trading.orders;
export const selectPositions = (state: any) => state.trading.positions;
export const selectMarketData = (state: any) => state.trading.marketData;
export const selectWatchlist = (state: any) => state.trading.watchlist;
export const selectSelectedSymbol = (state: any) => state.trading.selectedSymbol;
export const selectTradingLoading = (state: any) => state.trading.isLoading;
export const selectTradingError = (state: any) => state.trading.error;

export const selectPendingOrders = (state: any) =>
  state.trading.orders.filter((order: Order) => order.status === 'pending');

export const selectFilledOrders = (state: any) =>
  state.trading.orders.filter((order: Order) => order.status === 'filled');

export const selectMarketDataForSymbol = (symbol: string) => (state: any) =>
  state.trading.marketData[symbol];

// 导出 reducer
export default tradingSlice.reducer;





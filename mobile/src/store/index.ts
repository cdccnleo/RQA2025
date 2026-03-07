/**
 * RQA 2.0 Redux Store 配置
 *
 * 基于Redux Toolkit的现代化状态管理：
 * - 用户认证状态
 * - 投资组合数据
 * - 交易状态
 * - UI状态管理
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {configureStore, combineReducers} from '@reduxjs/toolkit';
import {
  persistStore,
  persistReducer,
  FLUSH,
  REHYDRATE,
  PAUSE,
  PERSIST,
  PURGE,
  REGISTER,
} from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Reducers
import authReducer from './slices/authSlice';
import portfolioReducer from './slices/portfolioSlice';
import tradingReducer from './slices/tradingSlice';
import strategiesReducer from './slices/strategiesSlice';

// Root reducer
const rootReducer = combineReducers({
  auth: authReducer,
  portfolio: portfolioReducer,
  trading: tradingReducer,
  strategies: strategiesReducer,
});

// Persist 配置
const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
  whitelist: ['auth', 'portfolio', 'ui'], // 只持久化这些状态
  blacklist: ['trading'], // 不持久化交易状态
};

// 持久化 reducer
const persistedReducer = persistReducer(persistConfig, rootReducer);

// Store 配置
export const store = configureStore({
  reducer: persistedReducer,
  middleware: getDefaultMiddleware =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
      },
    }),
  devTools: __DEV__, // 只在开发模式下启用 Redux DevTools
});

// Persistor
export const persistor = persistStore(store);

// 类型导出
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

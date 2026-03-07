/**
 * RQA 2.0 认证状态管理
 *
 * 处理用户登录、注册、登出、令牌管理等认证相关状态
 * 支持生物识别、记住密码等功能
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';
import AsyncStorage from '@react-native-async-storage/async-storage';

// 类型定义
interface User {
  id: string;
  email: string;
  username: string;
  firstName: string;
  lastName: string;
  avatar?: string;
  role: 'user' | 'premium' | 'admin';
  createdAt: string;
  lastLoginAt: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  biometricEnabled: boolean;
  rememberMe: boolean;
}

// 初始状态
const initialState: AuthState = {
  user: null,
  token: null,
  refreshToken: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  biometricEnabled: false,
  rememberMe: false,
};

// 异步操作
export const loginUser = createAsyncThunk(
  'auth/login',
  async (
    credentials: {email: string; password: string; rememberMe?: boolean},
    {rejectWithValue},
  ) => {
    try {
      // 模拟API调用
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        throw new Error('登录失败');
      }

      const data = await response.json();

      // 保存令牌到本地存储
      if (credentials.rememberMe) {
        await AsyncStorage.setItem('auth_token', data.token);
        await AsyncStorage.setItem('refresh_token', data.refreshToken);
      }

      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const registerUser = createAsyncThunk(
  'auth/register',
  async (
    userData: {
      email: string;
      password: string;
      username: string;
      firstName: string;
      lastName: string;
    },
    {rejectWithValue},
  ) => {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        throw new Error('注册失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

export const refreshToken = createAsyncThunk(
  'auth/refresh',
  async (refreshToken: string, {rejectWithValue}) => {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({refreshToken}),
      });

      if (!response.ok) {
        throw new Error('令牌刷新失败');
      }

      const data = await response.json();
      return data;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  },
);

// Auth Slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    logout: state => {
      state.user = null;
      state.token = null;
      state.refreshToken = null;
      state.isAuthenticated = false;
      state.error = null;
      // 清除本地存储
      AsyncStorage.removeItem('auth_token');
      AsyncStorage.removeItem('refresh_token');
    },
    clearError: state => {
      state.error = null;
    },
    setBiometricEnabled: (state, action: PayloadAction<boolean>) => {
      state.biometricEnabled = action.payload;
    },
    setRememberMe: (state, action: PayloadAction<boolean>) => {
      state.rememberMe = action.payload;
    },
    updateUser: (state, action: PayloadAction<Partial<User>>) => {
      if (state.user) {
        state.user = {...state.user, ...action.payload};
      }
    },
  },
  extraReducers: builder => {
    // 登录
    builder
      .addCase(loginUser.pending, state => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
        state.isAuthenticated = true;
        state.rememberMe = action.meta.arg.rememberMe || false;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // 注册
    builder
      .addCase(registerUser.pending, state => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
        state.isAuthenticated = true;
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });

    // 令牌刷新
    builder
      .addCase(refreshToken.fulfilled, (state, action) => {
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
      })
      .addCase(refreshToken.rejected, state => {
        // 令牌刷新失败，登出用户
        state.user = null;
        state.token = null;
        state.refreshToken = null;
        state.isAuthenticated = false;
        AsyncStorage.removeItem('auth_token');
        AsyncStorage.removeItem('refresh_token');
      });
  },
});

// 导出 actions
export const {
  logout,
  clearError,
  setBiometricEnabled,
  setRememberMe,
  updateUser,
} = authSlice.actions;

// 导出 reducer
export default authSlice.reducer;





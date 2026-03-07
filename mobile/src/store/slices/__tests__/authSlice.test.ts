/**
 * RQA 2.0 认证Slice测试
 *
 * 测试认证相关的状态管理、异步操作和reducer逻辑
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import authReducer, {
  loginUser,
  registerUser,
  logout,
  clearError,
  setBiometricEnabled,
  updateUser,
} from '../authSlice';

// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
  setItem: jest.fn(),
  getItem: jest.fn(),
  removeItem: jest.fn(),
}));

describe('Auth Slice', () => {
  const initialState = {
    user: null,
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
    biometricEnabled: false,
    rememberMe: false,
  };

  it('should return the initial state', () => {
    expect(authReducer(undefined, {type: undefined})).toEqual(initialState);
  });

  describe('logout', () => {
    it('should clear user data and tokens', () => {
      const loggedInState = {
        ...initialState,
        user: {id: '1', email: 'test@example.com'},
        token: 'test-token',
        refreshToken: 'test-refresh-token',
        isAuthenticated: true,
      };

      const action = logout();
      const result = authReducer(loggedInState, action);

      expect(result.user).toBeNull();
      expect(result.token).toBeNull();
      expect(result.refreshToken).toBeNull();
      expect(result.isAuthenticated).toBe(false);
    });
  });

  describe('clearError', () => {
    it('should clear error message', () => {
      const stateWithError = {
        ...initialState,
        error: 'Test error',
      };

      const action = clearError();
      const result = authReducer(stateWithError, action);

      expect(result.error).toBeNull();
    });
  });

  describe('setBiometricEnabled', () => {
    it('should set biometric enabled status', () => {
      const action = setBiometricEnabled(true);
      const result = authReducer(initialState, action);

      expect(result.biometricEnabled).toBe(true);
    });
  });

  describe('updateUser', () => {
    it('should update user data', () => {
      const stateWithUser = {
        ...initialState,
        user: {id: '1', email: 'test@example.com', firstName: 'John'},
      };

      const action = updateUser({firstName: 'Jane', lastName: 'Doe'});
      const result = authReducer(stateWithUser, action);

      expect(result.user).toEqual({
        id: '1',
        email: 'test@example.com',
        firstName: 'Jane',
        lastName: 'Doe',
      });
    });
  });

  describe('loginUser', () => {
    it('should handle loginUser.pending', () => {
      const action = {type: loginUser.pending.type};
      const result = authReducer(initialState, action);

      expect(result.isLoading).toBe(true);
      expect(result.error).toBeNull();
    });

    it('should handle loginUser.fulfilled', () => {
      const mockUser = {
        id: '1',
        email: 'test@example.com',
        username: 'testuser',
        firstName: 'John',
        lastName: 'Doe',
      };

      const action = {
        type: loginUser.fulfilled.type,
        payload: {
          user: mockUser,
          token: 'test-token',
          refreshToken: 'test-refresh-token',
        },
        meta: {
          arg: {rememberMe: true},
        },
      };

      const result = authReducer(
        {...initialState, isLoading: true},
        action
      );

      expect(result.isLoading).toBe(false);
      expect(result.user).toEqual(mockUser);
      expect(result.token).toBe('test-token');
      expect(result.refreshToken).toBe('test-refresh-token');
      expect(result.isAuthenticated).toBe(true);
      expect(result.rememberMe).toBe(true);
    });

    it('should handle loginUser.rejected', () => {
      const action = {
        type: loginUser.rejected.type,
        error: {message: 'Login failed'},
      };

      const result = authReducer(
        {...initialState, isLoading: true},
        action
      );

      expect(result.isLoading).toBe(false);
      expect(result.error).toBe('Login failed');
    });
  });

  describe('registerUser', () => {
    it('should handle registerUser.fulfilled', () => {
      const mockUser = {
        id: '1',
        email: 'new@example.com',
        username: 'newuser',
        firstName: 'Jane',
        lastName: 'Smith',
      };

      const action = {
        type: registerUser.fulfilled.type,
        payload: {
          user: mockUser,
          token: 'new-token',
          refreshToken: 'new-refresh-token',
        },
      };

      const result = authReducer(
        {...initialState, isLoading: true},
        action
      );

      expect(result.isLoading).toBe(false);
      expect(result.user).toEqual(mockUser);
      expect(result.isAuthenticated).toBe(true);
    });
  });
});





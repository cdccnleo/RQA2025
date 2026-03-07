/**
 * RQA 2.0 主题提供者
 *
 * 提供主题上下文，支持深色/浅色主题切换
 * 自动适应系统主题偏好
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React, {createContext, useContext, useState, useEffect} from 'react';
import {Appearance, ColorSchemeName} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// 主题配置
import {Theme, defaultTheme, darkTheme} from './theme';

// 主题上下文类型
interface ThemeContextType {
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

// 创建主题上下文
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// 主题模式
type ThemeMode = 'light' | 'dark' | 'system';

// ThemeProvider 组件
interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({children}) => {
  const [themeMode, setThemeMode] = useState<ThemeMode>('system');
  const [systemTheme, setSystemTheme] = useState<ColorSchemeName>(
    Appearance.getColorScheme(),
  );

  // 监听系统主题变化
  useEffect(() => {
    const subscription = Appearance.addChangeListener(({colorScheme}) => {
      setSystemTheme(colorScheme);
    });

    return () => subscription?.remove();
  }, []);

  // 初始化时加载保存的主题设置
  useEffect(() => {
    const loadThemePreference = async () => {
      try {
        const savedTheme = await AsyncStorage.getItem('theme_mode');
        if (savedTheme && ['light', 'dark', 'system'].includes(savedTheme)) {
          setThemeMode(savedTheme as ThemeMode);
        }
      } catch (error) {
        console.warn('Failed to load theme preference:', error);
      }
    };

    loadThemePreference();
  }, []);

  // 保存主题设置
  const saveThemePreference = async (mode: ThemeMode) => {
    try {
      await AsyncStorage.setItem('theme_mode', mode);
    } catch (error) {
      console.warn('Failed to save theme preference:', error);
    }
  };

  // 获取当前主题
  const getCurrentTheme = (): Theme => {
    let isDarkMode = false;

    switch (themeMode) {
      case 'light':
        isDarkMode = false;
        break;
      case 'dark':
        isDarkMode = true;
        break;
      case 'system':
      default:
        isDarkMode = systemTheme === 'dark';
        break;
    }

    return isDarkMode ? darkTheme : defaultTheme;
  };

  // 切换主题
  const toggleTheme = () => {
    const newMode: ThemeMode = themeMode === 'light' ? 'dark' : 'light';
    setThemeMode(newMode);
    saveThemePreference(newMode);
  };

  // 设置主题
  const setTheme = (mode: ThemeMode) => {
    setThemeMode(mode);
    saveThemePreference(mode);
  };

  // 当前主题
  const theme = getCurrentTheme();
  const isDark = theme === darkTheme;

  const contextValue: ThemeContextType = {
    theme,
    isDark,
    toggleTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

// 自定义 Hook
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export default ThemeProvider;





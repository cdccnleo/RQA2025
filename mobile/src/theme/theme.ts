/**
 * RQA 2.0 设计系统主题配置
 *
 * 统一的颜色、字体、间距、圆角等设计令牌
 * 支持深色/浅色主题切换
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

// 颜色系统
export const colors = {
  // 基础颜色
  primary: '#007AFF',
  secondary: '#5856D6',
  success: '#34C759',
  warning: '#FF9500',
  error: '#FF3B30',
  info: '#5AC8FA',

  // 背景颜色
  background: {
    primary: '#FFFFFF',
    secondary: '#F2F2F7',
    tertiary: '#E5E5EA',
  },

  // 文本颜色
  text: {
    primary: '#1C1C1E',
    secondary: '#8E8E93',
    tertiary: '#C7C7CC',
    inverse: '#FFFFFF',
  },

  // 边框颜色
  border: {
    light: '#E5E5EA',
    medium: '#C7C7CC',
    dark: '#8E8E93',
  },

  // 阴影颜色
  shadow: 'rgba(0, 0, 0, 0.1)',

  // 渐变色
  gradient: {
    primary: ['#007AFF', '#5856D6'],
    success: ['#34C759', '#30D158'],
    background: ['#1a1a2e', '#16213e', '#0f3460'],
  },
};

// 字体系统
export const typography = {
  // 字体族
  fontFamily: {
    regular: 'System',
    medium: 'System',
    bold: 'System',
    logo: 'System',
  },

  // 字体大小
  fontSize: {
    xs: 12,
    sm: 14,
    md: 16,
    lg: 18,
    xl: 20,
    xxl: 24,
    xxxl: 32,
    logo: 48,
  },

  // 行高
  lineHeight: {
    xs: 16,
    sm: 20,
    md: 24,
    lg: 28,
    xl: 32,
    xxl: 36,
    xxxl: 48,
  },

  // 字体权重
  fontWeight: {
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
  },

  // 预定义文本样式
  logo: {
    fontFamily: 'System',
    fontWeight: 'bold',
  },

  h1: {
    fontSize: 24,
    fontWeight: 'bold',
    lineHeight: 32,
  },

  h2: {
    fontSize: 20,
    fontWeight: 'bold',
    lineHeight: 28,
  },

  h3: {
    fontSize: 18,
    fontWeight: 'semibold',
    lineHeight: 24,
  },

  body: {
    fontSize: 16,
    fontWeight: 'normal',
    lineHeight: 24,
  },

  caption: {
    fontSize: 12,
    fontWeight: 'normal',
    lineHeight: 16,
  },

  button: {
    fontSize: 16,
    fontWeight: 'semibold',
    lineHeight: 24,
  },
};

// 间距系统
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
  xxxl: 64,
};

// 圆角系统
export const borderRadius = {
  xs: 2,
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  xxl: 24,
  round: 9999,
};

// 阴影系统
export const shadows = {
  sm: {
    shadowColor: colors.shadow,
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
    elevation: 3,
  },
  md: {
    shadowColor: colors.shadow,
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  lg: {
    shadowColor: colors.shadow,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 10,
  },
  xl: {
    shadowColor: colors.shadow,
    shadowOffset: {
      width: 0,
      height: 8,
    },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 20,
  },
};

// 动画时长
export const animation = {
  fast: 200,
  normal: 300,
  slow: 500,
};

// 断点 (用于响应式设计)
export const breakpoints = {
  phone: 0,
  tablet: 768,
  desktop: 1024,
};

// 主题类型定义
export interface Theme {
  colors: typeof colors;
  typography: typeof typography;
  spacing: typeof spacing;
  borderRadius: typeof borderRadius;
  shadows: typeof shadows;
  animation: typeof animation;
  breakpoints: typeof breakpoints;

  // 便捷访问
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  info: string;
}

// 默认主题
export const defaultTheme: Theme = {
  ...colors,
  ...typography,
  ...spacing,
  ...borderRadius,
  ...shadows,
  ...animation,
  ...breakpoints,
  colors,
  typography,
  spacing,
  borderRadius,
  shadows,
  animation,
  breakpoints,
};

// 深色主题
export const darkTheme: Theme = {
  ...defaultTheme,
  colors: {
    ...colors,
    background: {
      primary: '#1C1C1E',
      secondary: '#2C2C2E',
      tertiary: '#3A3A3C',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#EBEBF5',
      tertiary: '#C7C7CC',
      inverse: '#1C1C1E',
    },
    border: {
      light: '#3A3A3C',
      medium: '#48484A',
      dark: '#636366',
    },
  },
};





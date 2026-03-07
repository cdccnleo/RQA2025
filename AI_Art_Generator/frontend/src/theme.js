// AI Art Generator 主题配置

export const theme = {
  colors: {
    // 主色调 - 艺术紫色系
    primary: '#8B5CF6',
    primaryDark: '#7C3AED',
    primaryLight: '#A78BFA',

    // 辅助色
    secondary: '#06B6D4',
    secondaryDark: '#0891B2',
    accent: '#F59E0B',

    // 背景色
    background: '#0F0F23',
    surface: '#1A1A2E',
    surfaceLight: '#16213E',
    overlay: 'rgba(15, 15, 35, 0.8)',

    // 文字颜色
    text: '#FFFFFF',
    textSecondary: '#B8C5D1',
    textMuted: '#6B7B8C',

    // 状态色
    success: '#10B981',
    warning: '#F59E0B',
    error: '#EF4444',
    info: '#3B82F6',

    // 艺术色彩
    artistic: {
      purple: '#8B5CF6',
      blue: '#06B6D4',
      pink: '#EC4899',
      orange: '#F97316',
      green: '#22C55E',
      indigo: '#6366F1'
    }
  },

  fonts: {
    primary: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', monospace",
    artistic: "'Playfair Display', serif"
  },

  fontSizes: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '1.875rem',
    '4xl': '2.25rem',
    '5xl': '3rem'
  },

  spacing: {
    1: '0.25rem',
    2: '0.5rem',
    3: '0.75rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    8: '2rem',
    10: '2.5rem',
    12: '3rem',
    16: '4rem',
    20: '5rem',
    24: '6rem'
  },

  borderRadius: {
    none: '0',
    sm: '0.125rem',
    base: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    '3xl': '1.5rem',
    full: '9999px'
  },

  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    artistic: '0 25px 50px -12px rgba(139, 92, 246, 0.25)',
    glow: '0 0 20px rgba(139, 92, 246, 0.3)'
  },

  transitions: {
    fast: '0.15s ease',
    base: '0.3s ease',
    slow: '0.5s ease',
    bounce: '0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55)'
  },

  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px'
  },

  zIndex: {
    dropdown: 1000,
    sticky: 1020,
    fixed: 1030,
    modal: 1040,
    popover: 1050,
    tooltip: 1060
  }
};

// 艺术风格预设
export const artStyles = {
  random: {
    name: '随机生成',
    description: '基于噪声的自由创作',
    color: theme.colors.artistic.purple
  },
  abstract: {
    name: '抽象艺术',
    description: '现代抽象艺术风格',
    color: theme.colors.artistic.blue
  },
  impressionist: {
    name: '印象派',
    description: '莫奈风格的印象派',
    color: theme.colors.artistic.pink
  },
  cubist: {
    name: '立体派',
    description: '毕加索风格的立体主义',
    color: theme.colors.artistic.orange
  },
  surrealist: {
    name: '超现实主义',
    description: '达利风格的超现实',
    color: theme.colors.artistic.green
  },
  minimalist: {
    name: '极简主义',
    description: '现代极简艺术风格',
    color: theme.colors.artistic.indigo
  }
};

// 生成参数配置
export const generationParams = {
  quality: {
    standard: { size: 64, steps: 100 },
    high: { size: 128, steps: 200 },
    ultra: { size: 256, steps: 500 }
  },
  styles: artStyles
};


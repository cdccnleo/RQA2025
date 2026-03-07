import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { format, formatDistanceToNow, parseISO } from "date-fns"
import { zhCN } from "date-fns/locale"
import numeral from "numeral"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// 格式化数字
export function formatNumber(value: number, format: string = "0,0.00"): string {
  return numeral(value).format(format)
}

// 格式化百分比
export function formatPercent(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`
}

// 格式化货币
export function formatCurrency(value: number, currency: string = "CNY"): string {
  const symbols = {
    CNY: "¥",
    USD: "$",
    EUR: "€",
    GBP: "£",
    JPY: "¥",
  }

  const symbol = symbols[currency as keyof typeof symbols] || currency
  return `${symbol}${formatNumber(value)}`
}

// 格式化日期时间
export function formatDate(date: Date | string, formatStr: string = "yyyy-MM-dd HH:mm:ss"): string {
  const dateObj = typeof date === "string" ? parseISO(date) : date
  return format(dateObj, formatStr)
}

// 相对时间格式化
export function formatRelativeTime(date: Date | string): string {
  const dateObj = typeof date === "string" ? parseISO(date) : date
  return formatDistanceToNow(dateObj, { addSuffix: true, locale: zhCN })
}

// 截断文本
export function truncateText(text: string, maxLength: number = 50): string {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + "..."
}

// 生成随机ID
export function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36)
}

// 深度克隆对象
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") return obj
  if (obj instanceof Date) return new Date(obj.getTime()) as T
  if (obj instanceof Array) return obj.map(item => deepClone(item)) as T
  if (typeof obj === "object") {
    const clonedObj = {} as T
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key])
      }
    }
    return clonedObj
  }
  return obj
}

// 防抖函数
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout
  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

// 节流函数
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}

// 本地存储工具
export const storage = {
  get: <T>(key: string, defaultValue?: T): T | null => {
    if (typeof window === "undefined") return defaultValue || null
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : (defaultValue || null)
    } catch {
      return defaultValue || null
    }
  },

  set: <T>(key: string, value: T): void => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Ignore storage errors
    }
  },

  remove: (key: string): void => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.removeItem(key)
    } catch {
      // Ignore storage errors
    }
  },

  clear: (): void => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.clear()
    } catch {
      // Ignore storage errors
    }
  },
}

// 会话存储工具
export const sessionStorage = {
  get: <T>(key: string, defaultValue?: T): T | null => {
    if (typeof window === "undefined") return defaultValue || null
    try {
      const item = window.sessionStorage.getItem(key)
      return item ? JSON.parse(item) : (defaultValue || null)
    } catch {
      return defaultValue || null
    }
  },

  set: <T>(key: string, value: T): void => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Ignore storage errors
    }
  },

  remove: (key: string): void => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.removeItem(key)
    } catch {
      // Ignore storage errors
    }
  },

  clear: (): void => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.clear()
    } catch {
      // Ignore storage errors
    }
  },
}

// URL参数解析
export function getUrlParams(url?: string): Record<string, string> {
  if (typeof window === "undefined") return {}

  const searchParams = new URLSearchParams(
    url ? new URL(url).search : window.location.search
  )

  const params: Record<string, string> = {}
  searchParams.forEach((value, key) => {
    params[key] = value
  })

  return params
}

// 下载文件
export function downloadFile(data: Blob | string, filename: string, mimeType?: string): void {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mimeType || "text/plain" })

  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// 复制到剪贴板
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof window === "undefined") return false

  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Fallback for older browsers
    try {
      const textArea = document.createElement("textarea")
      textArea.value = text
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand("copy")
      document.body.removeChild(textArea)
      return true
    } catch {
      return false
    }
  }
}

// 检测设备类型
export const device = {
  isMobile: (): boolean => {
    if (typeof window === "undefined") return false
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent
    )
  },

  isTablet: (): boolean => {
    if (typeof window === "undefined") return false
    return /iPad|Android(?=.*\bMobile\b)|Tablet|PlayBook/i.test(navigator.userAgent)
  },

  isDesktop: (): boolean => {
    if (typeof window === "undefined") return true
    return !device.isMobile() && !device.isTablet()
  },

  isIOS: (): boolean => {
    if (typeof window === "undefined") return false
    return /iPad|iPhone|iPod/.test(navigator.userAgent)
  },

  isAndroid: (): boolean => {
    if (typeof window === "undefined") return false
    return /Android/.test(navigator.userAgent)
  },
}

// 性能监控
export const performance = {
  mark: (name: string): void => {
    if (typeof window !== "undefined" && window.performance) {
      window.performance.mark(name)
    }
  },

  measure: (name: string, startMark: string, endMark: string): void => {
    if (typeof window !== "undefined" && window.performance) {
      try {
        window.performance.measure(name, startMark, endMark)
      } catch {
        // Ignore measurement errors
      }
    }
  },

  getEntriesByName: (name: string): PerformanceEntry[] => {
    if (typeof window !== "undefined" && window.performance) {
      return window.performance.getEntriesByName(name)
    }
    return []
  },
}

// 错误处理
export class AppError extends Error {
  constructor(
    message: string,
    public code?: string,
    public statusCode?: number,
    public details?: any
  ) {
    super(message)
    this.name = "AppError"
  }
}

export function handleError(error: unknown): AppError {
  if (error instanceof AppError) {
    return error
  }

  if (error instanceof Error) {
    return new AppError(error.message)
  }

  return new AppError("An unknown error occurred", "UNKNOWN_ERROR")
}

// 类型守卫
export function isObject(value: any): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

export function isArray(value: any): value is any[] {
  return Array.isArray(value)
}

export function isString(value: any): value is string {
  return typeof value === "string"
}

export function isNumber(value: any): value is number {
  return typeof value === "number" && !isNaN(value)
}

export function isBoolean(value: any): value is boolean {
  return typeof value === "boolean"
}

export function isFunction(value: any): value is Function {
  return typeof value === "function"
}

// 数组工具
export const array = {
  unique: <T>(arr: T[]): T[] => [...new Set(arr)],

  chunk: <T>(arr: T[], size: number): T[][] => {
    const chunks: T[][] = []
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size))
    }
    return chunks
  },

  shuffle: <T>(arr: T[]): T[] => {
    const shuffled = [...arr]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  },

  groupBy: <T, K extends string | number | symbol>(
    arr: T[],
    keyFn: (item: T) => K
  ): Record<K, T[]> => {
    return arr.reduce((groups, item) => {
      const key = keyFn(item)
      if (!groups[key]) {
        groups[key] = []
      }
      groups[key].push(item)
      return groups
    }, {} as Record<K, T[]>)
  },
}

// 对象工具
export const object = {
  pick: <T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> => {
    const result = {} as Pick<T, K>
    keys.forEach(key => {
      if (key in obj) {
        result[key] = obj[key]
      }
    })
    return result
  },

  omit: <T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> => {
    const result = { ...obj }
    keys.forEach(key => {
      delete result[key]
    })
    return result
  },

  mapKeys: <T>(
    obj: Record<string, T>,
    mapper: (key: string, value: T) => string
  ): Record<string, T> => {
    return Object.entries(obj).reduce((result, [key, value]) => {
      result[mapper(key, value)] = value
      return result
    }, {} as Record<string, T>)
  },

  mapValues: <T, U>(
    obj: Record<string, T>,
    mapper: (value: T, key: string) => U
  ): Record<string, U> => {
    return Object.entries(obj).reduce((result, [key, value]) => {
      result[key] = mapper(value, key)
      return result
    }, {} as Record<string, U>)
  },
}


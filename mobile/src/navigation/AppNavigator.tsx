/**
 * RQA 2.0 移动端应用导航器
 *
 * 实现应用的主要导航结构：
 * - 认证流程 (登录/注册/忘记密码)
 * - 主应用界面 (底部标签导航)
 * - 模态界面 (设置/详情/交易)
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialIcons';

// 类型定义
import {RootStackParamList, MainTabParamList} from './types';

// 认证相关屏幕
import WelcomeScreen from '../screens/auth/WelcomeScreen';
import LoginScreen from '../screens/auth/LoginScreen';
import RegisterScreen from '../screens/auth/RegisterScreen';
import ForgotPasswordScreen from '../screens/auth/ForgotPasswordScreen';

// 主要功能屏幕
import PortfolioScreen from '../screens/main/PortfolioScreen';
import StrategiesScreen from '../screens/main/StrategiesScreen';
import TradingScreen from '../screens/main/TradingScreen';
import AnalyticsScreen from '../screens/main/AnalyticsScreen';
import ProfileScreen from '../screens/main/ProfileScreen';

// 模态屏幕
import StrategyDetailScreen from '../screens/modal/StrategyDetailScreen';
import TradeExecutionScreen from '../screens/modal/TradeExecutionScreen';
import SettingsScreen from '../screens/modal/SettingsScreen';

// 导航器实例
const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

/**
 * 主标签导航器
 */
const MainTabNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={({route}) => ({
        tabBarIcon: ({focused, color, size}) => {
          let iconName: string;

          switch (route.name) {
            case 'Portfolio':
              iconName = 'account-balance';
              break;
            case 'Strategies':
              iconName = 'analytics';
              break;
            case 'Trading':
              iconName = 'swap-horiz';
              break;
            case 'Analytics':
              iconName = 'bar-chart';
              break;
            case 'Profile':
              iconName = 'person';
              break;
            default:
              iconName = 'help';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: '#8E8E93',
        tabBarStyle: {
          backgroundColor: '#FFFFFF',
          borderTopColor: '#E5E5EA',
          borderTopWidth: 1,
          paddingBottom: 5,
          paddingTop: 5,
          height: 60,
        },
        headerShown: false,
      })}>
      <Tab.Screen
        name="Portfolio"
        component={PortfolioScreen}
        options={{
          title: '投资组合',
        }}
      />
      <Tab.Screen
        name="Strategies"
        component={StrategiesScreen}
        options={{
          title: '策略',
        }}
      />
      <Tab.Screen
        name="Trading"
        component={TradingScreen}
        options={{
          title: '交易',
        }}
      />
      <Tab.Screen
        name="Analytics"
        component={AnalyticsScreen}
        options={{
          title: '分析',
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          title: '我的',
        }}
      />
    </Tab.Navigator>
  );
};

/**
 * 主应用导航器
 */
const AppNavigator: React.FC = () => {
  return (
    <Stack.Navigator
      initialRouteName="Welcome"
      screenOptions={{
        headerStyle: {
          backgroundColor: '#1a1a2e',
        },
        headerTintColor: '#FFFFFF',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
        animation: 'slide_from_right',
      }}>
      {/* 认证流程 */}
      <Stack.Screen
        name="Welcome"
        component={WelcomeScreen}
        options={{headerShown: false}}
      />
      <Stack.Screen
        name="Login"
        component={LoginScreen}
        options={{
          title: '登录',
          headerBackTitle: '返回',
        }}
      />
      <Stack.Screen
        name="Register"
        component={RegisterScreen}
        options={{
          title: '注册',
          headerBackTitle: '返回',
        }}
      />
      <Stack.Screen
        name="ForgotPassword"
        component={ForgotPasswordScreen}
        options={{
          title: '忘记密码',
          headerBackTitle: '返回',
        }}
      />

      {/* 主应用界面 */}
      <Stack.Screen
        name="Main"
        component={MainTabNavigator}
        options={{headerShown: false}}
      />

      {/* 模态界面 */}
      <Stack.Screen
        name="StrategyDetail"
        component={StrategyDetailScreen}
        options={{
          title: '策略详情',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="TradeExecution"
        component={TradeExecutionScreen}
        options={{
          title: '执行交易',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{
          title: '设置',
          presentation: 'modal',
        }}
      />
    </Stack.Navigator>
  );
};

export default AppNavigator;





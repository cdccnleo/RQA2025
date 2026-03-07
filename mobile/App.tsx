/**
 * RQA2025 Mobile App
 * 量化交易系统移动端应用入口
 */

import React, {useEffect} from 'react';
import {Provider} from 'react-redux';
import {NavigationContainer} from '@react-navigation/native';
import {createStackNavigator} from '@react-navigation/stack';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

// Store
import {store} from './src/store';

// Screens
import HomeScreen from './src/screens/HomeScreen';
import SignalsScreen from './src/screens/SignalsScreen';
import PortfolioScreen from './src/screens/PortfolioScreen';
import MarketScreen from './src/screens/MarketScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import LoginScreen from './src/screens/LoginScreen';

// Services
import {initializePushNotifications} from './src/services/pushNotification';
import {initializeBiometrics} from './src/services/biometrics';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

// 主Tab导航
function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({route}) => ({
        tabBarIcon: ({focused, color, size}) => {
          let iconName: string;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Signals':
              iconName = focused ? 'signal' : 'signal-cellular-3';
              break;
            case 'Portfolio':
              iconName = focused ? 'chart-pie' : 'chart-pie-outline';
              break;
            case 'Market':
              iconName = focused ? 'chart-line' : 'chart-line-variant';
              break;
            case 'Settings':
              iconName = focused ? 'cog' : 'cog-outline';
              break;
            default:
              iconName = 'circle';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      })}>
      <Tab.Screen name="Home" component={HomeScreen} options={{title: '首页'}} />
      <Tab.Screen name="Signals" component={SignalsScreen} options={{title: '信号'}} />
      <Tab.Screen name="Portfolio" component={PortfolioScreen} options={{title: '组合'}} />
      <Tab.Screen name="Market" component={MarketScreen} options={{title: '行情'}} />
      <Tab.Screen name="Settings" component={SettingsScreen} options={{title: '设置'}} />
    </Tab.Navigator>
  );
}

// 主应用组件
function App(): React.JSX.Element {
  useEffect(() => {
    // 初始化推送通知
    initializePushNotifications();
    
    // 初始化生物识别
    initializeBiometrics();
  }, []);

  return (
    <SafeAreaProvider>
      <Provider store={store}>
        <NavigationContainer>
          <Stack.Navigator screenOptions={{headerShown: false}}>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Main" component={MainTabs} />
          </Stack.Navigator>
        </NavigationContainer>
      </Provider>
    </SafeAreaProvider>
  );
}

export default App;

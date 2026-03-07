/**
 * RQA 2.0 欢迎屏幕
 *
 * 用户首次打开应用时的欢迎界面
 * 展示产品特色，引导用户进行注册或登录
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  Image,
  SafeAreaView,
} from 'react-native';
import {useNavigation} from '@react-navigation/native';
import {NativeStackNavigationProp} from '@react-navigation/native-stack';
import LinearGradient from 'react-native-linear-gradient';

// 导航类型
import {RootStackParamList} from '../../navigation/types';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius} from '../../theme/theme';

const {width, height} = Dimensions.get('window');

type WelcomeScreenNavigationProp = NativeStackNavigationProp<
  RootStackParamList,
  'Welcome'
>;

const WelcomeScreen: React.FC = () => {
  const navigation = useNavigation<WelcomeScreenNavigationProp>();
  const {theme} = useTheme();

  const handleLogin = () => {
    navigation.navigate('Login');
  };

  const handleRegister = () => {
    navigation.navigate('Register');
  };

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient
        colors={['#1a1a2e', '#16213e', '#0f3460']}
        style={styles.gradient}>
        {/* Logo 和标题区域 */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Text style={styles.logoText}>RQA</Text>
            <Text style={styles.logoSubtitle}>2.0</Text>
          </View>
          <Text style={styles.title}>智能量化投资平台</Text>
          <Text style={styles.subtitle}>
            让专业级的量化投资策略触手可及
          </Text>
        </View>

        {/* 特色功能展示 */}
        <View style={styles.features}>
          <View style={styles.featureItem}>
            <View style={styles.featureIcon}>
              <Text style={styles.featureIconText}>🤖</Text>
            </View>
            <Text style={styles.featureTitle}>AI驱动策略</Text>
            <Text style={styles.featureDescription}>
              基于深度学习的智能投资策略
            </Text>
          </View>

          <View style={styles.featureItem}>
            <View style={styles.featureIcon}>
              <Text style={styles.featureIconText}>📊</Text>
            </View>
            <Text style={styles.featureTitle}>实时分析</Text>
            <Text style={styles.featureDescription}>
              全市场数据实时分析和风险评估
            </Text>
          </View>

          <View style={styles.featureItem}>
            <View style={styles.featureIcon}>
              <Text style={styles.featureIconText}>⚡</Text>
            </View>
            <Text style={styles.featureTitle}>极速交易</Text>
            <Text style={styles.featureDescription}>
              毫秒级订单执行，智能路由优化
            </Text>
          </View>
        </View>

        {/* 行动按钮 */}
        <View style={styles.actions}>
          <TouchableOpacity
            style={[styles.primaryButton, {backgroundColor: theme.primary}]}
            onPress={handleRegister}
            activeOpacity={0.8}>
            <Text style={styles.primaryButtonText}>开始使用</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={handleLogin}
            activeOpacity={0.6}>
            <Text style={styles.secondaryButtonText}>已有账号？登录</Text>
          </TouchableOpacity>
        </View>

        {/* 底部信息 */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>
            通过注册即表示您同意我们的
            <Text style={styles.linkText}> 服务条款 </Text>
            和
            <Text style={styles.linkText}> 隐私政策</Text>
          </Text>
        </View>
      </LinearGradient>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
    paddingHorizontal: spacing.lg,
  },
  header: {
    alignItems: 'center',
    marginTop: height * 0.1,
    marginBottom: height * 0.05,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: spacing.md,
  },
  logoText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: colors.primary,
    ...typography.logo,
  },
  logoSubtitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.secondary,
    marginLeft: spacing.xs,
    ...typography.logo,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
    ...typography.h1,
  },
  subtitle: {
    fontSize: 16,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: 24,
    ...typography.body,
  },
  features: {
    flex: 1,
    justifyContent: 'center',
    marginBottom: height * 0.05,
  },
  featureItem: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  featureIcon: {
    width: 64,
    height: 64,
    borderRadius: borderRadius.xl,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  featureIconText: {
    fontSize: 32,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: spacing.xs,
    ...typography.h3,
  },
  featureDescription: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: 20,
    ...typography.body,
  },
  actions: {
    marginBottom: height * 0.05,
  },
  primaryButton: {
    height: 56,
    borderRadius: borderRadius.lg,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
    shadowColor: '#007AFF',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  primaryButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.text.primary,
    ...typography.button,
  },
  secondaryButton: {
    height: 56,
    borderRadius: borderRadius.lg,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: colors.primary,
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.primary,
    ...typography.button,
  },
  footer: {
    alignItems: 'center',
    paddingBottom: spacing.xl,
  },
  footerText: {
    fontSize: 12,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: 18,
    ...typography.caption,
  },
  linkText: {
    color: colors.primary,
    textDecorationLine: 'underline',
  },
});

export default WelcomeScreen;





/**
 * RQA 2.0 快速交易模态框
 *
 * 提供快速的买入/卖出操作界面
 * 支持市价单和限价单，包含风险提示
 *
 * 作者: AI Assistant
 * 创建时间: 2025年12月4日
 */

import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  TextInput,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {useDispatch, useSelector} from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';

// Redux actions
import {placeOrder, selectMarketDataForSymbol} from '../../store/slices/tradingSlice';

// 主题和工具
import {useTheme} from '../../theme/ThemeProvider';
import {colors, typography, spacing, borderRadius} from '../../theme/theme';

interface QuickTradeModalProps {
  visible: boolean;
  symbol: string | null;
  onClose: () => void;
}

const QuickTradeModal: React.FC<QuickTradeModalProps> = ({
  visible,
  symbol,
  onClose,
}) => {
  const dispatch = useDispatch();
  const {theme} = useTheme();

  // 本地状态
  const [orderType, setOrderType] = useState<'buy' | 'sell'>('buy');
  const [orderMode, setOrderMode] = useState<'market' | 'limit'>('market');
  const [quantity, setQuantity] = useState('');
  const [limitPrice, setLimitPrice] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 获取市场数据
  const marketData = useSelector(symbol ? selectMarketDataForSymbol(symbol) : () => null);

  // 重置表单
  const resetForm = () => {
    setQuantity('');
    setLimitPrice('');
    setOrderMode('market');
  };

  // 关闭时重置
  useEffect(() => {
    if (!visible) {
      resetForm();
    }
  }, [visible]);

  // 提交订单
  const handleSubmitOrder = async () => {
    if (!symbol || !quantity) {
      Alert.alert('错误', '请输入交易数量');
      return;
    }

    const qty = parseFloat(quantity);
    if (isNaN(qty) || qty <= 0) {
      Alert.alert('错误', '请输入有效的交易数量');
      return;
    }

    // 限价单验证
    if (orderMode === 'limit') {
      if (!limitPrice) {
        Alert.alert('错误', '请输入限价');
        return;
      }
      const price = parseFloat(limitPrice);
      if (isNaN(price) || price <= 0) {
        Alert.alert('错误', '请输入有效的限价');
        return;
      }
    }

    setIsSubmitting(true);

    try {
      const orderData = {
        symbol,
        type: orderMode,
        side: orderType,
        quantity: qty,
        ...(orderMode === 'limit' && {price: parseFloat(limitPrice)}),
      };

      await dispatch(placeOrder(orderData) as any);

      Alert.alert(
        '订单提交成功',
        `${orderType === 'buy' ? '买入' : '卖出'} ${symbol} ${qty} 股的订单已提交`,
        [{text: '确定', onPress: () => {
          onClose();
        }}]
      );
    } catch (error) {
      Alert.alert('订单提交失败', '请稍后重试');
    } finally {
      setIsSubmitting(false);
    }
  };

  // 计算预估金额
  const calculateEstimatedAmount = () => {
    const qty = parseFloat(quantity);
    if (isNaN(qty) || qty <= 0) return 0;

    let price = marketData?.price || 0;
    if (orderMode === 'limit' && limitPrice) {
      price = parseFloat(limitPrice);
    }

    return qty * price;
  };

  if (!symbol) return null;

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
      transparent={false}>

      <KeyboardAvoidingView
        style={{flex: 1}}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>

        <View style={[styles.container, {backgroundColor: theme.colors.background.primary}]}>
          {/* 头部 */}
          <View style={styles.header}>
            <TouchableOpacity onPress={onClose} style={styles.closeButton}>
              <Icon name="close" size={24} color={colors.text.secondary} />
            </TouchableOpacity>

            <View style={styles.titleContainer}>
              <Text style={[styles.title, {color: theme.colors.text.primary}]}>
                {symbol}
              </Text>
              {marketData && (
                <Text style={[styles.subtitle, {color: theme.colors.text.secondary}]}>
                  ¥{marketData.price.toFixed(2)}
                </Text>
              )}
            </View>
          </View>

          {/* 交易类型选择 */}
          <View style={styles.tradeTypeContainer}>
            <TouchableOpacity
              style={[
                styles.tradeTypeButton,
                orderType === 'buy' && styles.buyTypeActive,
                {borderColor: colors.success}
              ]}
              onPress={() => setOrderType('buy')}>
              <Icon name="add" size={20} color={orderType === 'buy' ? colors.text.primary : colors.success} />
              <Text style={[
                styles.tradeTypeText,
                orderType === 'buy' && {color: colors.text.primary}
              ]}>
                买入
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.tradeTypeButton,
                orderType === 'sell' && styles.sellTypeActive,
                {borderColor: colors.error}
              ]}
              onPress={() => setOrderType('sell')}>
              <Icon name="remove" size={20} color={orderType === 'sell' ? colors.text.primary : colors.error} />
              <Text style={[
                styles.tradeTypeText,
                orderType === 'sell' && {color: colors.text.primary}
              ]}>
                卖出
              </Text>
            </TouchableOpacity>
          </View>

          {/* 订单类型选择 */}
          <View style={styles.orderModeContainer}>
            <TouchableOpacity
              style={[
                styles.orderModeButton,
                orderMode === 'market' && {backgroundColor: theme.primary}
              ]}
              onPress={() => setOrderMode('market')}>
              <Text style={[
                styles.orderModeText,
                orderMode === 'market' && {color: colors.text.primary}
              ]}>
                市价单
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.orderModeButton,
                orderMode === 'limit' && {backgroundColor: theme.primary}
              ]}
              onPress={() => setOrderMode('limit')}>
              <Text style={[
                styles.orderModeText,
                orderMode === 'limit' && {color: colors.text.primary}
              ]}>
                限价单
              </Text>
            </TouchableOpacity>
          </View>

          {/* 交易参数 */}
          <View style={styles.parametersContainer}>
            <View style={styles.parameterItem}>
              <Text style={[styles.parameterLabel, {color: theme.colors.text.primary}]}>
                数量
              </Text>
              <TextInput
                style={[styles.parameterInput, {color: theme.colors.text.primary}]}
                placeholder="输入股数"
                placeholderTextColor={colors.text.secondary}
                keyboardType="numeric"
                value={quantity}
                onChangeText={setQuantity}
              />
            </View>

            {orderMode === 'limit' && (
              <View style={styles.parameterItem}>
                <Text style={[styles.parameterLabel, {color: theme.colors.text.primary}]}>
                  限价
                </Text>
                <TextInput
                  style={[styles.parameterInput, {color: theme.colors.text.primary}]}
                  placeholder="输入价格"
                  placeholderTextColor={colors.text.secondary}
                  keyboardType="decimal-pad"
                  value={limitPrice}
                  onChangeText={setLimitPrice}
                />
              </View>
            )}
          </View>

          {/* 订单摘要 */}
          <View style={styles.summaryContainer}>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryLabel, {color: theme.colors.text.secondary}]}>
                预估金额
              </Text>
              <Text style={[styles.summaryValue, {color: theme.colors.text.primary}]}>
                ¥{calculateEstimatedAmount().toFixed(2)}
              </Text>
            </View>

            <View style={styles.summaryItem}>
              <Text style={[styles.summaryLabel, {color: theme.colors.text.secondary}]}>
                订单类型
              </Text>
              <Text style={[styles.summaryValue, {color: theme.colors.text.primary}]}>
                {orderMode === 'market' ? '市价' : '限价'} {orderType === 'buy' ? '买入' : '卖出'}
              </Text>
            </View>
          </View>

          {/* 风险提示 */}
          <View style={styles.riskWarning}>
            <Icon name="warning" size={16} color={colors.warning} />
            <Text style={[styles.riskWarningText, {color: theme.colors.text.secondary}]}>
              投资有风险，交易需谨慎。请确认订单信息后再提交。
            </Text>
          </View>

          {/* 操作按钮 */}
          <View style={styles.actions}>
            <TouchableOpacity
              style={[styles.cancelButton, {borderColor: colors.text.secondary}]}
              onPress={onClose}
              disabled={isSubmitting}>
              <Text style={[styles.cancelButtonText, {color: colors.text.secondary}]}>
                取消
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.submitButton,
                {backgroundColor: orderType === 'buy' ? colors.success : colors.error}
              ]}
              onPress={handleSubmitOrder}
              disabled={isSubmitting}>
              <Text style={styles.submitButtonText}>
                {isSubmitting ? '提交中...' : `确认${orderType === 'buy' ? '买入' : '卖出'}`}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.light,
  },
  closeButton: {
    padding: spacing.sm,
  },
  titleContainer: {
    flex: 1,
    alignItems: 'center',
  },
  title: {
    ...typography.h1,
    fontWeight: 'bold',
  },
  subtitle: {
    ...typography.body,
    marginTop: spacing.xs,
  },
  tradeTypeContainer: {
    flexDirection: 'row',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  tradeTypeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.lg,
    borderRadius: borderRadius.lg,
    borderWidth: 2,
    marginHorizontal: spacing.sm,
  },
  buyTypeActive: {
    backgroundColor: colors.success,
  },
  sellTypeActive: {
    backgroundColor: colors.error,
  },
  tradeTypeText: {
    ...typography.h3,
    fontWeight: 'bold',
    marginLeft: spacing.sm,
    color: colors.text.secondary,
  },
  orderModeContainer: {
    flexDirection: 'row',
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.lg,
  },
  orderModeButton: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    marginHorizontal: spacing.sm,
  },
  orderModeText: {
    ...typography.body,
    fontWeight: '600',
    color: colors.text.secondary,
  },
  parametersContainer: {
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.lg,
  },
  parameterItem: {
    marginBottom: spacing.md,
  },
  parameterLabel: {
    ...typography.body,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  parameterInput: {
    borderWidth: 1,
    borderColor: colors.border.light,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    ...typography.body,
  },
  summaryContainer: {
    backgroundColor: 'rgba(0, 122, 255, 0.1)',
    padding: spacing.lg,
    marginHorizontal: spacing.lg,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.lg,
  },
  summaryItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  summaryLabel: {
    ...typography.caption,
  },
  summaryValue: {
    ...typography.body,
    fontWeight: 'bold',
  },
  riskWarning: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.lg,
  },
  riskWarningText: {
    ...typography.caption,
    marginLeft: spacing.sm,
    flex: 1,
    lineHeight: 18,
  },
  actions: {
    flexDirection: 'row',
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
  },
  cancelButton: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.lg,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    marginRight: spacing.md,
  },
  cancelButtonText: {
    ...typography.button,
    fontWeight: '600',
  },
  submitButton: {
    flex: 2,
    alignItems: 'center',
    paddingVertical: spacing.lg,
    borderRadius: borderRadius.lg,
  },
  submitButtonText: {
    ...typography.button,
    color: colors.text.primary,
    fontWeight: 'bold',
  },
});

export default QuickTradeModal;





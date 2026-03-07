import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FiHome,
  FiImage,
  FiSettings,
  FiInfo,
  FiX,
  FiZap,
  FiPalette,
  FiTrendingUp
} from 'react-icons/fi';

const SidebarContainer = styled(motion.aside)`
  position: fixed;
  top: 70px;
  left: 0;
  width: 280px;
  height: calc(100vh - 70px);
  background: #1A1A2E;
  border-right: 1px solid rgba(139, 92, 246, 0.2);
  z-index: 999;
  overflow-y: auto;

  @media (max-width: 768px) {
    transform: ${props => props.isOpen ? 'translateX(0)' : 'translateX(-100%)'};
    transition: transform 0.3s ease;
  }
`;

const SidebarHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid rgba(139, 92, 246, 0.2);
  display: flex;
  align-items: center;
  justify-content: space-between;

  h2 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
`;

const CloseButton = styled(motion.button)`
  background: none;
  border: none;
  color: #B8C5D1;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 6px;
  transition: all 0.3s ease;

  &:hover {
    color: #FFFFFF;
    background: rgba(139, 92, 246, 0.1);
  }

  @media (min-width: 769px) {
    display: none;
  }
`;

const SidebarContent = styled.div`
  padding: 1rem;
`;

const MenuSection = styled.div`
  margin-bottom: 2rem;

  .section-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #B8C5D1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
    padding-left: 0.5rem;
  }
`;

const MenuItem = styled(motion.button)`
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: ${props => props.active ? 'rgba(139, 92, 246, 0.1)' : 'transparent'};
  border: 1px solid ${props => props.active ? 'rgba(139, 92, 246, 0.3)' : 'transparent'};
  border-radius: 8px;
  color: ${props => props.active ? '#8B5CF6' : '#B8C5D1'};
  text-align: left;
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: ${props => props.active ? '600' : '400'};
  transition: all 0.3s ease;
  margin-bottom: 0.25rem;

  &:hover {
    background: rgba(139, 92, 246, 0.05);
    border-color: rgba(139, 92, 246, 0.2);
    color: #FFFFFF;
    transform: translateX(4px);
  }

  svg {
    font-size: 1.1rem;
  }
`;

const StatsCard = styled(motion.div)`
  background: rgba(139, 92, 246, 0.05);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1.5rem;

  .stat-title {
    font-size: 0.875rem;
    color: #B8C5D1;
    margin-bottom: 0.5rem;
  }

  .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #8B5CF6;
    margin-bottom: 0.25rem;
  }

  .stat-desc {
    font-size: 0.8rem;
    color: #6B7B8C;
  }
`;

const RecentActivity = styled.div`
  .activity-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #B8C5D1;
    margin-bottom: 1rem;
    padding-left: 0.5rem;
  }
`;

const ActivityItem = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  margin-bottom: 0.5rem;
  border-left: 3px solid ${props => props.type === 'success' ? '#10B981' :
    props.type === 'warning' ? '#F59E0B' : '#8B5CF6'};

  .activity-icon {
    color: ${props => props.type === 'success' ? '#10B981' :
      props.type === 'warning' ? '#F59E0B' : '#8B5CF6'};
    font-size: 1rem;
  }

  .activity-content {
    flex: 1;

    .activity-text {
      font-size: 0.85rem;
      color: #FFFFFF;
      margin-bottom: 0.125rem;
    }

    .activity-time {
      font-size: 0.75rem;
      color: #6B7B8C;
    }
  }
`;

const menuItems = [
  {
    id: 'generator',
    icon: FiHome,
    label: '艺术生成器',
    description: 'AI艺术创作主界面'
  },
  {
    id: 'gallery',
    icon: FiImage,
    label: '作品画廊',
    description: '浏览和管理作品'
  },
  {
    id: 'analytics',
    icon: FiTrendingUp,
    label: '创作分析',
    description: '创作数据和趋势'
  },
  {
    id: 'settings',
    icon: FiSettings,
    label: '设置',
    description: '个性化设置'
  },
  {
    id: 'about',
    icon: FiInfo,
    label: '关于',
    description: '平台介绍和帮助'
  }
];

const Sidebar = ({ isOpen, onClose, currentView, onViewChange }) => {
  // 模拟统计数据
  const stats = {
    totalGenerations: 1247,
    uniqueUsers: 89,
    avgQuality: 4.2
  };

  // 模拟近期活动
  const recentActivities = [
    {
      id: 1,
      type: 'success',
      text: '成功生成4张抽象艺术作品',
      time: '2分钟前',
      icon: FiPalette
    },
    {
      id: 2,
      type: 'info',
      text: '用户反馈：生成速度很快',
      time: '5分钟前',
      icon: FiZap
    },
    {
      id: 3,
      type: 'warning',
      text: '检测到GPU内存使用率较高',
      time: '10分钟前',
      icon: FiTrendingUp
    }
  ];

  return (
    <AnimatePresence>
      {(isOpen) && (
        <>
          {/* 移动端遮罩 */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 70,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.5)',
              zIndex: 998,
              display: window.innerWidth <= 768 ? 'block' : 'none'
            }}
            onClick={onClose}
          />

          <SidebarContainer
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            isOpen={isOpen}
          >
            <SidebarHeader>
              <h2>
                <FiPalette />
                AI Art Generator
              </h2>
              <CloseButton
                onClick={onClose}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <FiX size={20} />
              </CloseButton>
            </SidebarHeader>

            <SidebarContent>
              {/* 平台统计 */}
              <StatsCard
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <div className="stat-title">今日生成</div>
                <div className="stat-value">{stats.totalGenerations.toLocaleString()}</div>
                <div className="stat-desc">张艺术作品</div>
              </StatsCard>

              {/* 导航菜单 */}
              <MenuSection>
                <div className="section-title">导航</div>
                {menuItems.map((item, index) => (
                  <MenuItem
                    key={item.id}
                    active={currentView === item.id}
                    onClick={() => onViewChange(item.id)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <item.icon />
                    <div>
                      <div>{item.label}</div>
                      <div style={{ fontSize: '0.75rem', color: '#6B7B8C', marginTop: '2px' }}>
                        {item.description}
                      </div>
                    </div>
                  </MenuItem>
                ))}
              </MenuSection>

              {/* 近期活动 */}
              <RecentActivity>
                <div className="activity-title">近期活动</div>
                {recentActivities.map((activity, index) => (
                  <ActivityItem
                    key={activity.id}
                    type={activity.type}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <activity.icon className="activity-icon" />
                    <div className="activity-content">
                      <div className="activity-text">{activity.text}</div>
                      <div className="activity-time">{activity.time}</div>
                    </div>
                  </ActivityItem>
                ))}
              </RecentActivity>
            </SidebarContent>
          </SidebarContainer>
        </>
      )}
    </AnimatePresence>
  );
};

export default Sidebar;


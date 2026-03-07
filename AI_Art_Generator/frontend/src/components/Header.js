import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FiMenu, FiZap, FiPalette } from 'react-icons/fi';

const HeaderContainer = styled(motion.header)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 70px;
  background: rgba(15, 15, 35, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(139, 92, 246, 0.2);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;

  @media (max-width: 768px) {
    padding: 0 1rem;
  }
`;

const Logo = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.5rem;
  font-weight: 700;
  color: #8B5CF6;
  cursor: pointer;

  span {
    background: linear-gradient(135deg, #8B5CF6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  @media (max-width: 768px) {
    font-size: 1.25rem;
    gap: 0.5rem;
  }
`;

const NavActions = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const MenuButton = styled(motion.button)`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  background: rgba(139, 92, 246, 0.1);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 8px;
  color: #8B5CF6;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(139, 92, 246, 0.2);
    border-color: rgba(139, 92, 246, 0.4);
    transform: translateY(-2px);
  }

  @media (max-width: 768px) {
    width: 40px;
    height: 40px;
  }
`;

const StatusIndicator = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: ${props => props.connected
    ? 'rgba(16, 185, 129, 0.1)'
    : 'rgba(239, 68, 68, 0.1)'};
  border: 1px solid ${props => props.connected
    ? 'rgba(16, 185, 129, 0.3)'
    : 'rgba(239, 68, 68, 0.3)'};
  border-radius: 20px;
  font-size: 0.875rem;
  color: ${props => props.connected ? '#10B981' : '#EF4444'};

  @media (max-width: 768px) {
    display: none;
  }
`;

const StatusDot = styled(motion.div)`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.connected ? '#10B981' : '#EF4444'};
`;

const Header = ({ onMenuClick, sidebarOpen }) => {
  // 模拟连接状态
  const [isConnected, setIsConnected] = React.useState(true);

  return (
    <HeaderContainer
      initial={{ y: -70 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <Logo
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FiPalette size={24} />
        <span>AI Art Generator</span>
      </Logo>

      <NavActions>
        <StatusIndicator connected={isConnected}>
          <StatusDot
            connected={isConnected}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [1, 0.7, 1]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <span>{isConnected ? 'AI引擎已连接' : '连接中...'}</span>
          <FiZap size={14} />
        </StatusIndicator>

        <MenuButton
          onClick={onMenuClick}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          animate={{ rotate: sidebarOpen ? 90 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <FiMenu size={20} />
        </MenuButton>
      </NavActions>
    </HeaderContainer>
  );
};

export default Header;


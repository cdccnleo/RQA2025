import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const Overlay = styled(motion.div)`
  position: ${props => props.fullScreen ? 'fixed' : 'absolute'};
  top: ${props => props.fullScreen ? 0 : 'auto'};
  left: ${props => props.fullScreen ? 0 : 'auto'};
  right: ${props => props.fullScreen ? 0 : 'auto'};
  bottom: ${props => props.fullScreen ? 0 : 'auto'};
  width: ${props => props.fullScreen ? '100%' : 'auto'};
  height: ${props => props.fullScreen ? '100vh' : 'auto'};
  background: ${props => props.fullScreen
    ? 'rgba(15, 15, 35, 0.95)'
    : 'rgba(26, 26, 46, 0.9)'};
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: ${props => props.fullScreen ? 2000 : 100};
  color: white;
  padding: ${props => props.fullScreen ? '2rem' : '1rem'};
  border-radius: ${props => props.fullScreen ? 0 : '8px'};
`;

const SpinnerContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
`;

const Spinner = styled(motion.div)`
  width: 60px;
  height: 60px;
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-top: 4px solid #8B5CF6;
  border-radius: 50%;
`;

const Content = styled.div`
  text-align: center;
  max-width: 400px;

  h2 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #FFFFFF;
  }

  p {
    font-size: 1rem;
    color: #B8C5D1;
    margin-bottom: 0;
    line-height: 1.5;
  }

  .loading-dots::after {
    content: '';
    animation: loadingDots 1.5s infinite;
  }

  @keyframes loadingDots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
  }
`;

const ProgressIndicator = styled(motion.div)`
  width: 200px;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 1rem;
`;

const ProgressBar = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #8B5CF6, #06B6D4);
  border-radius: 2px;
`;

const LoadingSpinner = ({
  message = "加载中...",
  fullScreen = false,
  showProgress = false,
  progress = 0
}) => {
  return (
    <Overlay
      fullScreen={fullScreen}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      <SpinnerContainer>
        <Spinner
          animate={{ rotate: 360 }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "linear"
          }}
        />

        <Content>
          <h2>{message}</h2>
          {showProgress && (
            <ProgressIndicator>
              <ProgressBar
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </ProgressIndicator>
          )}
        </Content>
      </SpinnerContainer>
    </Overlay>
  );
};

export default LoadingSpinner;


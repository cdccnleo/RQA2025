import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { FiDownload, FiRefreshCw, FiZap, FiSettings, FiImage, FiLoader } from 'react-icons/fi';
import { artStyles, generationParams } from '../theme';

const GeneratorContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding-top: 100px; /* 为固定header留出空间 */
`;

const HeroSection = styled(motion.section)`
  text-align: center;
  margin-bottom: 3rem;

  h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #8B5CF6, #06B6D4, #F59E0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  p {
    font-size: 1.25rem;
    color: #B8C5D1;
    margin-bottom: 2rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
  }

  @media (max-width: 768px) {
    h1 {
      font-size: 2.5rem;
    }

    p {
      font-size: 1.1rem;
    }
  }
`;

const ControlPanel = styled(motion.div)`
  background: #1A1A2E;
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  border: 1px solid rgba(139, 92, 246, 0.2);
  box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
`;

const ControlGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
`;

const ControlGroup = styled.div`
  .control-label {
    display: block;
    margin-bottom: 0.75rem;
    font-weight: 600;
    color: #FFFFFF;
    font-size: 0.95rem;
  }

  input, select {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 2px solid rgba(139, 92, 246, 0.2);
    border-radius: 8px;
    background: rgba(139, 92, 246, 0.05);
    color: #FFFFFF;
    font-size: 1rem;
    transition: all 0.3s ease;

    &:focus {
      outline: none;
      border-color: #8B5CF6;
      box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }

    &::placeholder {
      color: #6B7B8C;
    }
  }
`;

const StyleSelector = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 0.5rem;
`;

const StyleButton = styled(motion.button)`
  padding: 0.5rem 1rem;
  border: 2px solid ${props => props.selected ? props.color : 'rgba(139, 92, 246, 0.2)'};
  border-radius: 20px;
  background: ${props => props.selected ? props.color : 'transparent'};
  color: ${props => props.selected ? '#FFFFFF' : '#B8C5D1'};
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.3s ease;

  &:hover {
    border-color: ${props => props.color};
    transform: translateY(-2px);
  }
`;

const GenerateButton = styled(motion.button)`
  grid-column: 1 / -1;
  padding: 1rem 3rem;
  background: linear-gradient(135deg, #8B5CF6, #06B6D4);
  border: none;
  border-radius: 12px;
  color: white;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &:hover:not(:disabled) {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
  }

  &:hover:not(:disabled)::before {
    left: 100%;
  }
`;

const ResultsSection = styled(motion.section)`
  margin-top: 3rem;
`;

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 2rem;
  margin-top: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
`;

const ArtCard = styled(motion.div)`
  background: #1A1A2E;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(139, 92, 246, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2);
    border-color: rgba(139, 92, 246, 0.4);
  }
`;

const ArtImage = styled.div`
  position: relative;
  width: 100%;
  height: 280px;
  background: #16213E;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
  }

  ${ArtCard}:hover & img {
    transform: scale(1.05);
  }
`;

const ArtInfo = styled.div`
  padding: 1.5rem;

  h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
  }

  p {
    margin: 0 0 1rem 0;
    font-size: 0.9rem;
    color: #B8C5D1;
  }
`;

const ArtActions = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const ActionButton = styled(motion.button)`
  flex: 1;
  padding: 0.5rem;
  border: 1px solid rgba(139, 92, 246, 0.3);
  border-radius: 6px;
  background: transparent;
  color: #8B5CF6;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  font-size: 0.85rem;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(139, 92, 246, 0.1);
    border-color: #8B5CF6;
  }
`;

const LoadingOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(15, 15, 35, 0.9);
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  color: white;
`;

const ProgressBar = styled.div`
  width: 300px;
  height: 8px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
  margin: 2rem 0;

  @media (max-width: 768px) {
    width: 250px;
  }
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #8B5CF6, #06B6D4);
  border-radius: 4px;
`;

const ArtGenerator = () => {
  const [numImages, setNumImages] = useState(4);
  const [selectedStyle, setSelectedStyle] = useState('random');
  const [quality, setQuality] = useState('standard');
  const [seed, setSeed] = useState('');
  const [generatedImages, setGeneratedImages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);

  // 获取随机种子
  const getRandomSeed = useCallback(async () => {
    try {
      const response = await axios.get('/generate/random-seed');
      setSeed(response.data.seed.toString());
    } catch (error) {
      console.error('Failed to get random seed:', error);
      setSeed(Math.floor(Math.random() * 1000000).toString());
    }
  }, []);

  // 生成艺术
  const generateArt = useCallback(async () => {
    if (isGenerating) return;

    setIsGenerating(true);
    setGenerationProgress(0);
    setGeneratedImages([]);

    try {
      // 模拟进度更新
      const progressInterval = setInterval(() => {
        setGenerationProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const requestData = {
        num_images: numImages,
        style: selectedStyle,
        quality: quality,
        seed: seed ? parseInt(seed) : undefined
      };

      const response = await axios.post('/generate', requestData);

      clearInterval(progressInterval);
      setGenerationProgress(100);

      // 短暂延迟后显示结果
      setTimeout(() => {
        setGeneratedImages(response.data.images.map((imageData, index) => ({
          id: `art_${Date.now()}_${index}`,
          imageData,
          style: selectedStyle,
          quality,
          seed: seed || 'random',
          timestamp: new Date().toISOString()
        })));
        setIsGenerating(false);
        setGenerationProgress(0);
      }, 500);

    } catch (error) {
      console.error('Art generation failed:', error);
      setIsGenerating(false);
      setGenerationProgress(0);
      // 这里可以显示错误提示
    }
  }, [numImages, selectedStyle, quality, seed, isGenerating]);

  // 下载图像
  const downloadImage = useCallback((imageData, filename) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // 重新生成
  const regenerate = useCallback(() => {
    getRandomSeed();
  }, [getRandomSeed]);

  // 初始化时获取随机种子
  useEffect(() => {
    getRandomSeed();
  }, [getRandomSeed]);

  return (
    <GeneratorContainer>
      <HeroSection
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <h1>🎨 AI艺术创作平台</h1>
        <p>
          让每个人都能成为艺术家！基于深度学习的AI艺术生成引擎，
          支持多种艺术风格和创作模式。
        </p>
      </HeroSection>

      <ControlPanel
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <ControlGrid>
          <ControlGroup>
            <label className="control-label">生成数量</label>
            <select
              value={numImages}
              onChange={(e) => setNumImages(parseInt(e.target.value))}
              disabled={isGenerating}
            >
              <option value={1}>1张</option>
              <option value={4}>4张</option>
              <option value={9}>9张</option>
            </select>
          </ControlGroup>

          <ControlGroup>
            <label className="control-label">艺术风格</label>
            <StyleSelector>
              {Object.entries(artStyles).map(([key, style]) => (
                <StyleButton
                  key={key}
                  color={style.color}
                  selected={selectedStyle === key}
                  onClick={() => setSelectedStyle(key)}
                  disabled={isGenerating}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {style.name}
                </StyleButton>
              ))}
            </StyleSelector>
          </ControlGroup>

          <ControlGroup>
            <label className="control-label">画质设置</label>
            <select
              value={quality}
              onChange={(e) => setQuality(e.target.value)}
              disabled={isGenerating}
            >
              <option value="standard">标准 (64x64)</option>
              <option value="high">高清 (128x128)</option>
              <option value="ultra">超清 (256x256)</option>
            </select>
          </ControlGroup>

          <ControlGroup>
            <label className="control-label">
              随机种子
              <motion.button
                onClick={regenerate}
                disabled={isGenerating}
                style={{
                  marginLeft: '0.5rem',
                  background: 'none',
                  border: 'none',
                  color: '#8B5CF6',
                  cursor: 'pointer',
                  padding: '0.25rem'
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <FiRefreshCw size={16} />
              </motion.button>
            </label>
            <input
              type="number"
              placeholder="留空使用随机种子"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              disabled={isGenerating}
            />
          </ControlGroup>

          <GenerateButton
            onClick={generateArt}
            disabled={isGenerating}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isGenerating ? (
              <>
                <FiLoader className="loading-dots" />
                正在创作中...
              </>
            ) : (
              <>
                <FiZap />
                开始创作
              </>
            )}
          </GenerateButton>
        </ControlGrid>
      </ControlPanel>

      <AnimatePresence>
        {isGenerating && (
          <LoadingOverlay
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <h2>🎨 AI正在创作中...</h2>
              <p>正在运用深度学习生成您的艺术作品</p>
              <ProgressBar>
                <ProgressFill
                  initial={{ width: 0 }}
                  animate={{ width: `${generationProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </ProgressBar>
              <p>{generationProgress}% 完成</p>
            </motion.div>
          </LoadingOverlay>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {generatedImages.length > 0 && (
          <ResultsSection
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.h2
              initial={{ y: 20 }}
              animate={{ y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              🎨 创作结果
            </motion.h2>

            <ResultsGrid>
              {generatedImages.map((art, index) => (
                <ArtCard
                  key={art.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <ArtImage>
                    <img src={art.imageData} alt={`AI Art ${index + 1}`} />
                  </ArtImage>

                  <ArtInfo>
                    <h3>作品 #{index + 1}</h3>
                    <p>
                      风格: {artStyles[art.style]?.name || art.style}<br/>
                      画质: {generationParams.quality[art.quality]?.size || art.quality}x{generationParams.quality[art.quality]?.size || art.quality}<br/>
                      种子: {art.seed}
                    </p>

                    <ArtActions>
                      <ActionButton
                        onClick={() => downloadImage(art.imageData, `ai_art_${art.id}.png`)}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <FiDownload size={14} />
                        下载
                      </ActionButton>

                      <ActionButton
                        onClick={() => {/* TODO: 分享功能 */}}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <FiImage size={14} />
                        分享
                      </ActionButton>
                    </ArtActions>
                  </ArtInfo>
                </ArtCard>
              ))}
            </ResultsGrid>
          </ResultsSection>
        )}
      </AnimatePresence>
    </GeneratorContainer>
  );
};

export default ArtGenerator;


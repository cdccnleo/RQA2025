#!/bin/bash

# RQA2025 风险控制层构建脚本
# 支持多架构构建和优化

set -e

# 配置变量
REGISTRY="${REGISTRY:-rqa2025}"
IMAGE_NAME="${IMAGE_NAME:-risk-control}"
TAG="${TAG:-latest}"
BUILD_TYPE="${BUILD_TYPE:-production}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查构建依赖..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose 未安装，将跳过相关功能"
    fi

    log_success "依赖检查完成"
}

# 安全扫描
security_scan() {
    log_info "执行安全扫描..."

    if command -v trivy &> /dev/null; then
        log_info "使用 Trivy 进行容器安全扫描..."
        trivy image --exit-code 0 --no-progress --format json "${REGISTRY}/${IMAGE_NAME}:${TAG}" > security-report.json
        log_success "安全扫描完成，结果已保存到 security-report.json"
    else
        log_warning "Trivy 未安装，跳过安全扫描"
    fi
}

# 构建基础镜像
build_base_image() {
    log_info "构建基础镜像..."

    docker build \
        --target dependency-builder \
        --tag "${REGISTRY}/risk-control-base:${TAG}" \
        --cache-from "${REGISTRY}/risk-control-base:latest" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .

    log_success "基础镜像构建完成"
}

# 构建GPU镜像
build_gpu_image() {
    log_info "构建GPU镜像..."

    if ! docker buildx ls | grep -q multi-arch; then
        docker buildx create --name multi-arch --use
    fi

    docker buildx build \
        --platform linux/amd64 \
        --target gpu-production \
        --tag "${REGISTRY}/risk-control-gpu:${TAG}" \
        --cache-from "${REGISTRY}/risk-control-gpu:latest" \
        --load \
        .

    log_success "GPU镜像构建完成"
}

# 构建生产镜像
build_production_image() {
    log_info "构建生产镜像..."

    # 单架构构建
    if [ "$BUILD_TYPE" = "single" ]; then
        docker build \
            --target production \
            --tag "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
            --cache-from "${REGISTRY}/${IMAGE_NAME}:latest" \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            .

    # 多架构构建
    else
        if ! docker buildx ls | grep -q multi-arch; then
            docker buildx create --name multi-arch --use
        fi

        docker buildx build \
            --platform "$PLATFORMS" \
            --target production \
            --tag "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
            --cache-from "${REGISTRY}/${IMAGE_NAME}:latest" \
            --push \
            .
    fi

    log_success "生产镜像构建完成"
}

# 构建开发镜像
build_development_image() {
    log_info "构建开发镜像..."

    docker build \
        --target development \
        --tag "${REGISTRY}/${IMAGE_NAME}-dev:${TAG}" \
        --cache-from "${REGISTRY}/${IMAGE_NAME}-dev:latest" \
        .

    log_success "开发镜像构建完成"
}

# 构建监控镜像
build_monitoring_image() {
    log_info "构建监控镜像..."

    docker build \
        --target monitoring \
        --tag "${REGISTRY}/${IMAGE_NAME}-monitor:${TAG}" \
        --cache-from "${REGISTRY}/${IMAGE_NAME}-monitor:latest" \
        .

    log_success "监控镜像构建完成"
}

# 推送镜像
push_images() {
    log_info "推送镜像到注册表..."

    # 推送生产镜像
    docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"

    # 推送其他镜像
    if [ "$BUILD_TYPE" != "gpu-only" ]; then
        docker push "${REGISTRY}/risk-control-base:${TAG}"
        docker push "${REGISTRY}/${IMAGE_NAME}-dev:${TAG}"
        docker push "${REGISTRY}/${IMAGE_NAME}-monitor:${TAG}"
    fi

    # 推送GPU镜像
    if [ "$BUILD_TYPE" = "gpu" ] || [ "$BUILD_TYPE" = "all" ]; then
        docker push "${REGISTRY}/risk-control-gpu:${TAG}"
    fi

    log_success "镜像推送完成"
}

# 清理构建缓存
cleanup() {
    log_info "清理构建缓存..."

    # 删除悬空镜像
    docker image prune -f

    # 删除停止的容器
    docker container prune -f

    log_success "缓存清理完成"
}

# 显示构建信息
show_build_info() {
    echo "========================================"
    echo "RQA2025 风险控制层构建信息"
    echo "========================================"
    echo "注册表: $REGISTRY"
    echo "镜像名称: $IMAGE_NAME"
    echo "标签: $TAG"
    echo "构建类型: $BUILD_TYPE"
    echo "目标平台: $PLATFORMS"
    echo "========================================"
}

# 显示使用帮助
show_help() {
    echo "RQA2025 风险控制层构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示帮助信息"
    echo "  -r, --registry REG      指定镜像注册表 (默认: rqa2025)"
    echo "  -n, --name NAME         指定镜像名称 (默认: risk-control)"
    echo "  -t, --tag TAG           指定镜像标签 (默认: latest)"
    echo "  -b, --build-type TYPE   构建类型: production, gpu, dev, monitor, all (默认: production)"
    echo "  -p, --platforms PLAT    目标平台 (默认: linux/amd64,linux/arm64)"
    echo "  --push                  构建后推送镜像"
    echo "  --scan                  执行安全扫描"
    echo "  --cleanup               构建后清理缓存"
    echo ""
    echo "示例:"
    echo "  $0 --build-type all --push --scan"
    echo "  $0 -b gpu -t v2.0.0 --push"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -n|--name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            -b|--build-type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -p|--platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            --push)
                PUSH_IMAGES=true
                shift
                ;;
            --scan)
                SECURITY_SCAN=true
                shift
                ;;
            --cleanup)
                CLEANUP=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    parse_args "$@"
    show_build_info

    # 检查依赖
    check_dependencies

    # 根据构建类型执行相应的构建
    case $BUILD_TYPE in
        production)
            build_base_image
            build_production_image
            ;;
        gpu)
            build_gpu_image
            ;;
        dev)
            build_development_image
            ;;
        monitor)
            build_monitoring_image
            ;;
        all)
            build_base_image
            build_production_image
            build_gpu_image
            build_development_image
            build_monitoring_image
            ;;
        *)
            log_error "不支持的构建类型: $BUILD_TYPE"
            exit 1
            ;;
    esac

    # 推送镜像
    if [ "$PUSH_IMAGES" = true ]; then
        push_images
    fi

    # 安全扫描
    if [ "$SECURITY_SCAN" = true ]; then
        security_scan
    fi

    # 清理缓存
    if [ "$CLEANUP" = true ]; then
        cleanup
    fi

    log_success "构建完成！"
    echo ""
    echo "可用镜像:"
    if [ "$BUILD_TYPE" = "production" ] || [ "$BUILD_TYPE" = "all" ]; then
        echo "  - ${REGISTRY}/${IMAGE_NAME}:${TAG}"
    fi
    if [ "$BUILD_TYPE" = "gpu" ] || [ "$BUILD_TYPE" = "all" ]; then
        echo "  - ${REGISTRY}/risk-control-gpu:${TAG}"
    fi
    if [ "$BUILD_TYPE" = "dev" ] || [ "$BUILD_TYPE" = "all" ]; then
        echo "  - ${REGISTRY}/${IMAGE_NAME}-dev:${TAG}"
    fi
    if [ "$BUILD_TYPE" = "monitor" ] || [ "$BUILD_TYPE" = "all" ]; then
        echo "  - ${REGISTRY}/${IMAGE_NAME}-monitor:${TAG}"
    fi
}

# 执行主函数
main "$@"

# 导入路径优化报告

**优化时间**: 2025-07-19
**优化文件数**: 17

## 优化详情

### src\aliases.py
-   from src.acceleration.fpga import FpgaManager as FPGA
from src -> from .acceleration.fpga import FpgaManager as FPGA
from src

### src\adapters\miniqmt\adapter.py
-   from src.adapters.miniqmt.miniqmt_data_adapter import MiniQMTDataAdapter
from src -> from .adapters.miniqmt.miniqmt_data_adapter import MiniQMTDataAdapter
from src
-   from src.adapters.miniqmt.data_cache import MiniQMTDataCache
from src -> from .adapters.miniqmt.data_cache import MiniQMTDataCache
from src

### src\backtest\backtest_engine.py
-   from src.fpga.fpga_manager import FPGAManager

 -> from .fpga.fpga_manager import FPGAManager



### src\backtest\data_loader.py
-   from src.utils.date_utils import convert_timezone

logger  -> from .utils.date_utils import convert_timezone

logger 

### src\backtest\engine.py
-   from src.utils.logger import get_logger
from src -> from .utils.logger import get_logger
from src

### src\backtest\parameter_optimizer.py
-   from src.backtest.engine import BacktestEngine
from src -> from .backtest.engine import BacktestEngine
from src

### src\data\china\dragon_board_updater.py
-   from src.features import FeatureEngineer
        FeatureEngineer -> from .features import FeatureEngineer
        FeatureEngineer

### src\data\loader\batch_loader.py
-   from src.infrastructure import config
from  -> from .infrastructure import config
from 

### src\engine\stress_test.py
-   from src.utils.logger import get_logger
from src -> from .utils.logger import get_logger
from src

### src\features\signal_generator.py
-   from src.acceleration.fpga import FpgaManager

logger  -> from .acceleration.fpga import FpgaManager

logger 

### src\fpga\fpga_optimizer.py
-   from src.risk.risk_controller import RiskConfig

logger  -> from .risk.risk_controller import RiskConfig

logger 

### src\fpga\fpga_order_optimizer.py
-   from src.fpga.fpga_manager import FPGAManager
from src -> from .fpga.fpga_manager import FPGAManager
from src

### src\fpga\fpga_risk_engine.py
-   from src.fpga.fpga_manager import FPGAManager
from src -> from .fpga.fpga_manager import FPGAManager
from src

### src\fpga\fpga_sentiment_analyzer.py
-   from src.fpga.fpga_manager import FPGAManager
from src -> from .fpga.fpga_manager import FPGAManager
from src

### src\trading\risk\china\circuit_breaker.py
-   from src.fpga.fpga_risk_engine import FpgaRiskEngine
            self -> from .fpga.fpga_risk_engine import FpgaRiskEngine
            self

### src\trading\risk\china\star_market.py
-   from src.utils.logger import get_logger
from src -> from .utils.logger import get_logger
from src

### src\tuning\evaluators\early_stopping.py
-   from src.tuning.optimizers.base import ObjectiveDirection

class EarlyStopping -> from .tuning.optimizers.base import ObjectiveDirection

class EarlyStopping


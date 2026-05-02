# DermAgent 代码规范

## 1. 导入规范

### 1.1 导入顺序
按照以下顺序组织导入：
1. 标准库导入
2. 第三方库导入
3. 本地应用/库导入

```python
# 标准库
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 第三方库
import numpy as np
import pandas as pd
import torch
from PIL import Image

# 本地导入
from agent.state import AgentState
from skills.base import BaseSkill
from memory.experience_bank import ExperienceBank
```

### 1.2 相对导入 vs 绝对导入
- **推荐使用绝对导入**，从项目根目录开始
- 避免使用相对导入（`from . import`），除非在同一包内

```python
# 推荐 ✓
from agent.planner import Planner
from skills.catalog import SkillCatalog
from dataio.case_loader import CaseLoader

# 不推荐 ✗
from .planner import Planner
from ..skills.catalog import SkillCatalog
```

### 1.3 导入别名
- 使用标准别名：`np`, `pd`, `plt`
- 避免使用 `import *`

```python
# 推荐 ✓
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 不推荐 ✗
from numpy import *
```

## 2. 命名规范

### 2.1 文件和目录
- 使用小写字母和下划线：`case_loader.py`, `experience_bank.py`
- 避免使用连字符：`case-loader.py` ✗

### 2.2 类名
- 使用 PascalCase（大驼峰）：`CaseLoader`, `ExperienceBank`

```python
class CaseLoader:
    pass

class ExperienceBank:
    pass
```

### 2.3 函数和变量
- 使用 snake_case（小写+下划线）：`load_case()`, `experience_count`

```python
def load_case(case_id: str) -> Dict:
    experience_count = 0
    return {}
```

### 2.4 常量
- 使用全大写+下划线：`MAX_RETRIES`, `DEFAULT_TIMEOUT`

```python
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "http://localhost:8000"
```

### 2.5 私有成员
- 使用单下划线前缀：`_internal_method()`, `_cache`

```python
class MyClass:
    def __init__(self):
        self._cache = {}
    
    def _internal_method(self):
        pass
```

## 3. 类型注解

### 3.1 函数签名
- 所有公共函数应有类型注解
- 使用 `typing` 模块的类型

```python
from typing import List, Dict, Optional, Union

def process_cases(
    case_ids: List[str],
    config: Dict[str, any],
    timeout: Optional[int] = None
) -> List[Dict]:
    """处理多个案例"""
    return []
```

### 3.2 类属性
- 使用类型注解声明属性

```python
from typing import List, Dict

class CaseLoader:
    cases: List[Dict]
    config: Dict[str, any]
    
    def __init__(self):
        self.cases = []
        self.config = {}
```

## 4. 文档字符串

### 4.1 模块文档
- 每个模块开头应有文档字符串

```python
"""
Case Loader Module

This module provides functionality for loading and processing
dermatological cases from various datasets.
"""
```

### 4.2 类文档
- 使用 Google 风格或 NumPy 风格

```python
class CaseLoader:
    """
    Load and process dermatological cases.
    
    This class handles loading cases from different datasets,
    normalizing formats, and providing a unified interface.
    
    Attributes:
        data_root: Root directory for data files
        cache_enabled: Whether to enable caching
    
    Example:
        >>> loader = CaseLoader("/path/to/data")
        >>> case = loader.load("case_001")
    """
```

### 4.3 函数文档
```python
def load_case(case_id: str, include_metadata: bool = True) -> Dict:
    """
    Load a single case by ID.
    
    Args:
        case_id: Unique identifier for the case
        include_metadata: Whether to include metadata in result
    
    Returns:
        Dictionary containing case data and optionally metadata
    
    Raises:
        ValueError: If case_id is invalid
        FileNotFoundError: If case file doesn't exist
    
    Example:
        >>> case = load_case("case_001")
        >>> print(case['diagnosis'])
    """
```

## 5. 错误处理

### 5.1 异常类型
- 使用具体的异常类型
- 避免裸 `except:`

```python
# 推荐 ✓
try:
    result = process_case(case_id)
except FileNotFoundError as e:
    logger.error(f"Case file not found: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid case data: {e}")
    return None

# 不推荐 ✗
try:
    result = process_case(case_id)
except:
    pass
```

### 5.2 自定义异常
```python
class CaseLoadError(Exception):
    """Raised when case loading fails"""
    pass

class InvalidDiagnosisError(ValueError):
    """Raised when diagnosis format is invalid"""
    pass
```

## 6. 日志规范

### 6.1 日志级别
- DEBUG: 详细调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误

```python
import logging

logger = logging.getLogger(__name__)

logger.debug(f"Processing case {case_id}")
logger.info(f"Loaded {len(cases)} cases")
logger.warning(f"Case {case_id} has missing metadata")
logger.error(f"Failed to load case {case_id}: {error}")
logger.critical(f"System failure: {error}")
```

### 6.2 日志格式
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dermagent.log'),
        logging.StreamHandler()
    ]
)
```

## 7. 代码组织

### 7.1 文件结构
```python
"""Module docstring"""

# Imports
import os
from typing import List

# Constants
MAX_CASES = 1000

# Classes
class MyClass:
    pass

# Functions
def my_function():
    pass

# Main execution
if __name__ == "__main__":
    main()
```

### 7.2 类结构
```python
class MyClass:
    """Class docstring"""
    
    # Class variables
    class_var = "value"
    
    def __init__(self):
        """Initialize"""
        self.instance_var = "value"
    
    # Public methods
    def public_method(self):
        """Public method"""
        pass
    
    # Private methods
    def _private_method(self):
        """Private method"""
        pass
    
    # Special methods
    def __str__(self):
        return "MyClass"
```

## 8. 测试规范

### 8.1 测试文件命名
- 测试文件以 `test_` 开头：`test_case_loader.py`
- 测试类以 `Test` 开头：`TestCaseLoader`
- 测试函数以 `test_` 开头：`test_load_case()`

```python
# tests/test_case_loader.py
import pytest
from dataio.case_loader import CaseLoader

class TestCaseLoader:
    def test_load_case(self):
        loader = CaseLoader("/path/to/data")
        case = loader.load("case_001")
        assert case is not None
    
    def test_load_invalid_case(self):
        loader = CaseLoader("/path/to/data")
        with pytest.raises(ValueError):
            loader.load("invalid_id")
```

### 8.2 测试覆盖率
- 目标：80% 以上代码覆盖率
- 重点测试：核心逻辑、边界条件、错误处理

## 9. 性能优化

### 9.1 避免重复计算
```python
# 推荐 ✓
result = expensive_computation()
for item in items:
    process(item, result)

# 不推荐 ✗
for item in items:
    result = expensive_computation()
    process(item, result)
```

### 9.2 使用生成器
```python
# 推荐 ✓ - 内存高效
def load_cases_generator(case_ids):
    for case_id in case_ids:
        yield load_case(case_id)

# 不推荐 ✗ - 内存占用大
def load_cases_list(case_ids):
    return [load_case(case_id) for case_id in case_ids]
```

## 10. Git 提交规范

### 10.1 提交信息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 10.2 Type 类型
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 10.3 示例
```
feat(dataio): add SCIN dataset loader

- Implement SCINLoader class
- Add schema validation
- Update documentation

Closes #123
```

## 11. 代码审查清单

- [ ] 代码符合命名规范
- [ ] 有适当的类型注解
- [ ] 有完整的文档字符串
- [ ] 有错误处理
- [ ] 有日志记录
- [ ] 有单元测试
- [ ] 通过 Black 格式化
- [ ] 通过 flake8 检查
- [ ] 通过 mypy 类型检查
- [ ] 测试覆盖率 > 80%

## 12. 工具配置

### 12.1 Black
```bash
black --line-length 100 .
```

### 12.2 isort
```bash
isort --profile black .
```

### 12.3 flake8
```bash
flake8 --max-line-length 100 --extend-ignore E203,W503 .
```

### 12.4 mypy
```bash
mypy --ignore-missing-imports .
```

### 12.5 pytest
```bash
pytest --cov=. --cov-report=html --cov-report=term
```

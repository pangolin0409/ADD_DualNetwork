"""
MLflow 工具函數
用於整合 MLflow 實驗追蹤和模型管理
"""

import os
import yaml
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowManager:
    """MLflow 管理器類"""
    
    def __init__(self, config_path: str = "./config/mlflow.yml"):
        """
        初始化 MLflow 管理器
        
        Args:
            config_path: MLflow 配置文件路徑
        """
        self.config = self._load_config(config_path)
        self._setup_mlflow()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """載入 MLflow 配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('mlflow', {})
        except FileNotFoundError:
            logger.warning(f"MLflow 配置文件 {config_path} 不存在，使用默認配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取默認配置"""
        return {
            'tracking_uri': 'file:./mlruns',
            'experiment_name': 'audio-deepfake-detection',
            'autolog': {'enabled': True, 'log_models': True},
            'tags': {'project': 'ADD_DualNetwork', 'framework': 'PyTorch'}
        }
    
    def _setup_mlflow(self):
        """設置 MLflow"""
        # 設置追蹤 URI
        tracking_uri = self.config.get('tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # 設置實驗
        experiment_name = self.config.get('experiment_name', 'audio-deepfake-detection')
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"創建新實驗: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"使用現有實驗: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.error(f"設置實驗時發生錯誤: {e}")
            experiment_id = mlflow.create_experiment(experiment_name)
        
        self.experiment_id = experiment_id
        
        # 設置自動記錄
        autolog_config = self.config.get('autolog', {})
        if autolog_config.get('enabled', True):
            mlflow.pytorch.autolog(
                log_models=autolog_config.get('log_models', True),
                log_datasets=autolog_config.get('log_datasets', True)
            )
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        開始 MLflow 運行
        
        Args:
            run_name: 運行名稱
            tags: 自定義標籤
            
        Returns:
            MLflow ActiveRun 對象
        """
        # 合併默認標籤和自定義標籤
        default_tags = self.config.get('tags', {})
        if tags:
            default_tags.update(tags)
        
        # 添加時間戳標籤
        default_tags['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags
        )
    
    def log_params(self, params: Dict[str, Any]):
        """
        記錄參數
        
        Args:
            params: 參數字典
        """
        # 過濾掉 None 值
        filtered_params = {k: v for k, v in params.items() if v is not None}
        mlflow.log_params(filtered_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        記錄指標
        
        Args:
            metrics: 指標字典
            step: 步驟數（用於時間序列指標）
        """
        if step is not None:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metrics(metrics)
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model", 
                  input_example: Optional[torch.Tensor] = None,
                  signature: Optional[mlflow.models.signature.ModelSignature] = None):
        """
        記錄 PyTorch 模型
        
        Args:
            model: PyTorch 模型
            model_name: 模型名稱
            input_example: 輸入範例
            signature: 模型簽名
        """
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                input_example=input_example,
                signature=signature
            )
            logger.info(f"成功記錄模型: {model_name}")
        except Exception as e:
            logger.error(f"記錄模型時發生錯誤: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        記錄文件夾中的所有文件
        
        Args:
            local_dir: 本地文件夾路徑
            artifact_path: 在 MLflow 中的路徑
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"成功記錄文件夾: {local_dir}")
        except Exception as e:
            logger.error(f"記錄文件夾時發生錯誤: {e}")
    
    def log_figure(self, figure, artifact_file: str):
        """
        記錄圖表
        
        Args:
            figure: matplotlib 圖表對象
            artifact_file: 文件名
        """
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.info(f"成功記錄圖表: {artifact_file}")
        except Exception as e:
            logger.error(f"記錄圖表時發生錯誤: {e}")
    
    def log_confusion_matrix(self, confusion_matrix: np.ndarray, 
                           labels: List[str] = None, 
                           artifact_file: str = "confusion_matrix.json"):
        """
        記錄混淆矩陣
        
        Args:
            confusion_matrix: 混淆矩陣
            labels: 標籤列表
            artifact_file: 文件名
        """
        try:
            cm_data = {
                'confusion_matrix': confusion_matrix.tolist(),
                'labels': labels or ['Bona fide', 'Spoof'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(artifact_file, 'w') as f:
                json.dump(cm_data, f, indent=2)
            
            mlflow.log_artifact(artifact_file)
            os.remove(artifact_file)  # 清理臨時文件
            logger.info(f"成功記錄混淆矩陣: {artifact_file}")
        except Exception as e:
            logger.error(f"記錄混淆矩陣時發生錯誤: {e}")
    
    def register_model(self, model_name: str, model_version: str = None, 
                      stage: str = "None", description: str = ""):
        """
        註冊模型到模型註冊表
        
        Args:
            model_name: 模型名稱
            model_version: 模型版本
            stage: 模型階段 (None, Staging, Production, Archived)
            description: 模型描述
        """
        try:
            # 獲取當前運行的模型 URI
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            # 註冊模型
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )
            
            # 設置模型階段
            if stage != "None":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model.version,
                    stage=stage
                )
            
            logger.info(f"成功註冊模型: {model_name} (版本: {registered_model.version})")
            return registered_model
            
        except Exception as e:
            logger.error(f"註冊模型時發生錯誤: {e}")
            return None
    
    def compare_runs(self, run_ids: List[str], metrics: List[str] = None, 
                    params: List[str] = None) -> Dict[str, Any]:
        """
        比較多個運行
        
        Args:
            run_ids: 運行 ID 列表
            metrics: 要比較的指標列表
            params: 要比較的參數列表
            
        Returns:
            比較結果字典
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # 獲取默認比較項目
            if metrics is None:
                metrics = self.config.get('comparison', {}).get('metrics_to_compare', 
                                                               ['EER', 'accuracy', 'epoch_loss'])
            if params is None:
                params = self.config.get('comparison', {}).get('params_to_compare', 
                                                              ['lr', 'batch_size', 'num_epochs'])
            
            comparison_data = {}
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                comparison_data[run_id] = {
                    'metrics': {metric: run.data.metrics.get(metric) for metric in metrics},
                    'params': {param: run.data.params.get(param) for param in params},
                    'tags': run.data.tags
                }
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"比較運行時發生錯誤: {e}")
            return {}
    
    def get_best_run(self, metric_name: str = "EER", ascending: bool = True) -> Optional[str]:
        """
        獲取最佳運行
        
        Args:
            metric_name: 指標名稱
            ascending: 是否升序排列（True 表示越小越好）
            
        Returns:
            最佳運行的 ID
        """
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if runs:
                best_run = runs[0]
                logger.info(f"找到最佳運行: {best_run.info.run_id} ({metric_name}: {best_run.data.metrics.get(metric_name)})")
                return best_run.info.run_id
            else:
                logger.warning("沒有找到任何運行")
                return None
                
        except Exception as e:
            logger.error(f"獲取最佳運行時發生錯誤: {e}")
            return None


def create_model_signature(input_shape: tuple, output_shape: tuple) -> mlflow.models.signature.ModelSignature:
    """
    創建模型簽名
    
    Args:
        input_shape: 輸入形狀
        output_shape: 輸出形狀
        
    Returns:
        模型簽名
    """
    from mlflow.types.schema import Schema, TensorSpec
    
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), input_shape)
    ])
    
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32), output_shape)
    ])
    
    return mlflow.models.signature.ModelSignature(inputs=input_schema, outputs=output_schema)


def log_training_artifacts(mlflow_manager: MLflowManager, args, model_path: str, log_path: str):
    """
    記錄訓練相關的文件
    
    Args:
        mlflow_manager: MLflow 管理器
        args: 訓練參數
        model_path: 模型保存路徑
        log_path: 日誌路徑
    """
    try:
        # 記錄配置文件
        config_files = [
            "./config/base.yml",
            "./config/mlflow.yml",
            "./config/aug_group.yml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                mlflow.log_artifact(config_file, "config")
        
        # 記錄模型檢查點
        if os.path.exists(model_path):
            mlflow.log_artifacts(model_path, "checkpoints")
        
        # 記錄訓練日誌
        if os.path.exists(log_path):
            mlflow.log_artifacts(log_path, "logs")
        
        # 記錄代碼版本信息
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.object.hexsha)
            mlflow.log_param("git_branch", repo.active_branch.name)
        except:
            logger.warning("無法獲取 Git 信息")
        
        logger.info("成功記錄訓練相關文件")
        
    except Exception as e:
        logger.error(f"記錄訓練文件時發生錯誤: {e}")


# 便利函數
def get_mlflow_manager(config_path: str = "./config/mlflow.yml") -> MLflowManager:
    """獲取 MLflow 管理器實例"""
    return MLflowManager(config_path)


def log_audio_metrics(mlflow_manager: MLflowManager, metrics: Dict[str, float]):
    """
    記錄音訊相關指標
    
    Args:
        mlflow_manager: MLflow 管理器
        metrics: 指標字典
    """
    # 添加音訊特定的標籤
    audio_tags = {
        'task_type': 'audio_classification',
        'domain': 'deepfake_detection'
    }
    
    mlflow_manager.log_metrics(metrics)
    mlflow.set_tags(audio_tags)

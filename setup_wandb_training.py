import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

# wandb 로거 설정 예시
def setup_wandb_logger(project_name="stable-audio-cvae", experiment_name=None, config=None):
  """
  Wandb 로거를 설정하는 함수
  
  Args:
    project_name: wandb 프로젝트 이름
    experiment_name: 실험 이름 (None이면 자동 생성)
    config: 실험 설정 딕셔너리
  
  Returns:
    WandbLogger: 설정된 wandb 로거
  """
  wandb_logger = WandbLogger(
    project=project_name,
    name=experiment_name,
    config=config,
    save_dir="./wandb_logs",
    log_model=True,  # 모델 체크포인트를 wandb에 업로드
  )
  
  return wandb_logger

# 학습 설정 예시
def train_with_wandb():
  """
  Wandb 로거를 사용한 학습 예시
  """
  # 실험 설정
  config = {
    "lr": 1e-4,
    "batch_size": 16,
    "sample_rate": 48000,
    "warmup_steps": 1000,
    "use_ema": True,
    # 다른 하이퍼파라미터들...
  }
  
  # wandb 로거 설정
  wandb_logger = setup_wandb_logger(
    project_name="stable-audio-cvae",
    experiment_name="autoencoder_training_v1",
    config=config
  )
  
  # 모델 초기화 (예시)
  # autoencoder = AudioAutoencoder(...)
  # model = AutoencoderTrainingWrapper(autoencoder, **config)
  
  # 트레이너 설정
  trainer = pl.Trainer(
    max_epochs=100,
    logger=wandb_logger,  # wandb 로거 연결
    accelerator="gpu",
    devices=1,
    precision=16,  # mixed precision training
    gradient_clip_val=0.5,
    val_check_interval=0.25,  # validation 체크 간격
    log_every_n_steps=50,     # 로깅 간격
    callbacks=[
      # 체크포인트 콜백
      pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        filename="{epoch}-{step}-{val_loss:.3f}",
        monitor="val/pesq" if "pesq" in config else "val/stft",
        mode="max" if "pesq" in config else "min",
        save_top_k=3,
        save_last=True
      ),
      # 조기 종료 콜백
      pl.callbacks.EarlyStopping(
        monitor="val/pesq" if "pesq" in config else "val/stft",
        mode="max" if "pesq" in config else "min",
        patience=10,
        min_delta=0.001
      ),
      # 학습률 모니터링
      pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
  )
  
  # 학습 시작
  # trainer.fit(model, train_dataloader, val_dataloader)
  
  # wandb 종료
  wandb.finish()

if __name__ == "__main__":
  # wandb 로그인 (최초 1회만 필요)
  # wandb login
  
  train_with_wandb() 
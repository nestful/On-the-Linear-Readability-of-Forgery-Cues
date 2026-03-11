import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import json
from datetime import datetime
import logging
import torch.cuda.amp as amp

# ==================== 日志工具 ====================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 日志文件名加上 nolora
    log_file = os.path.join(log_dir, f'training_nolora_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__), log_file

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.json_log_file = os.path.join(log_dir, f'metrics_nolora_{timestamp}.json')
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 'val_auc': [], 
            'learning_rates': []
        }
    
    def log_epoch(self, epoch, metrics):
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(float(v))
        with open(self.json_log_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_final(self, best_epoch, best_acc):
        results = {
            'experiment': 'No-LoRA (Frozen Backbone)',
            'best_epoch': best_epoch,
            'best_val_acc': float(best_acc),
            'training_history': self.history
        }
        # 保存最终结果概要
        with open(os.path.join(self.log_dir, 'final_results_nolora.json'), 'w') as f:
            json.dump(results, f, indent=4)

# ==================== 数据集定义 ====================
class ForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        valid_folders = ['0_real', '1_fake']
        
        # print(f"Scanning images in {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root)
            if folder_name in valid_folders:
                label = int(folder_name.split('_')[0])
                for img_name in files:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.samples.append(os.path.join(root, img_name))
                        self.labels.append(label)
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {root_dir}.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ==================== 模型定义 (冻结骨干) ====================
class DINOv2FrozenBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading DINOv2 (Frozen Backbone)...")
        # 加载预训练模型
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        # 核心：冻结骨干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.feature_dim = 1024
        
        # 只训练这个分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1)
        )
    
    def forward(self, x):
        # 使用 no_grad 确保 backbone 不计算梯度
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)
    
    def get_trainable_params(self):
        return self.classifier.parameters()

# ==================== 训练与验证 ====================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, logger):
    model.train()
    total_loss = 0
    preds_list, labels_list = [], []
    
    pbar = tqdm(loader, desc="Training (Head Only)", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        with amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = (logits > 0).float()
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    acc = accuracy_score(labels_list, preds_list)
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device, logger):
    model.eval()
    total_loss = 0
    preds_list, labels_list, probs_list = [], [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            probs_list.extend(probs.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader),
        'acc': accuracy_score(labels_list, preds_list),
        'auc': roc_auc_score(labels_list, probs_list) if len(set(labels_list)) > 1 else 0.5
    }
    logger.info(f"Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['acc']:.4f} | Val AUC: {metrics['auc']:.4f}")
    return metrics

# ==================== 主程序 ====================
def main():
    CFG = {
        'batch_size': 32,
        'lr': 5e-5,           # Linear probing 学习率稍大
        'epochs': 10,       # 保持和原实验一致，或者减少
        'train_path': '/opt/data/private/data/train',
        'val_path': '/opt/data/private/data/val',
        'log_dir': './logs1',
        'ckpt_dir': './checkpoints' # 保存路径
    }
    
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    logger, log_file = setup_logger(CFG['log_dir'])
    train_logger = TrainingLogger(CFG['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*60)
    logger.info("ABLATION STUDY: No-LoRA (Frozen Backbone)")
    logger.info(f"Saving models to: {CFG['ckpt_dir']}/*_nolora.pth")
    logger.info("="*60)
    
    # 增强与加载
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(
        ForgeryDataset(CFG['train_path'], transform=train_transform),
        batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        ForgeryDataset(CFG['val_path'], transform=val_transform),
        batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 模型初始化
    model = DINOv2FrozenBaseline().to(device)
    
    # 优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.get_trainable_params(), lr=CFG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = amp.GradScaler()
    
    # 定义模型保存路径
    best_acc = 0.0
    best_epoch = 0
    # 这里的命名加上了 _nolora
    best_model_path = os.path.join(CFG['ckpt_dir'], 'best_model_nolora_linear.pth')
    
    logger.info("Starting Training...")
    
    for epoch in range(CFG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, logger
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device, logger)
        
        scheduler.step(val_metrics['loss'])
        
        # 记录
        train_logger.log_epoch(epoch+1, {
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
            'val_auc': val_metrics['auc'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型 (带 nolora 后缀)
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✓ Best model saved to {best_model_path} (Acc: {best_acc:.4f})")
        
        # 定期保存 (带 nolora 后缀)
        if (epoch + 1) % 10 == 0:
            ckpt_name = f'checkpoint_epoch_{epoch+1}_nolora.pth'
            ckpt_path = os.path.join(CFG['ckpt_dir'], ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc']
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_name}")

    logger.info("\n" + "="*60)
    logger.info("Training Completed (No Testing Phase).")
    logger.info(f"Best Val Acc: {best_acc:.4f} at Epoch {best_epoch}")
    train_logger.save_final(best_epoch, best_acc)

if __name__ == '__main__':
    main()
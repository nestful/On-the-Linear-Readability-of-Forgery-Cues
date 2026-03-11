import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json
from datetime import datetime
import logging
import torch.cuda.amp as amp

# ==================== LoRA实现 ====================
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

def inject_lora_to_linear(model, rank=8, alpha=16, target_modules=['qkv', 'proj','mlp']):
    lora_params = []
    for name, module in model.named_modules():
        # 检查模块名中是否包含目标字符串 (例如 'mlp' 会匹配 'blocks.0.mlp.fc1')
        is_target = any(target in name for target in target_modules)
        if isinstance(module, nn.Linear) and is_target:
            in_features = module.in_features
            out_features = module.out_features
            
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
            lora_layer = LoRALayer(in_features, out_features, rank, alpha)
            original_forward = module.forward
            
            def make_lora_forward(orig_forward, lora):
                def forward(x):
                    return orig_forward(x) + lora(x)
                return forward
            
            module.forward = make_lora_forward(original_forward, lora_layer)
            module.lora_layer = lora_layer
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
    
    return lora_params

# ==================== 日志工具 ====================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
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
        self.json_log_file = os.path.join(log_dir, f'metrics_{timestamp}.json')
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
    
    def save_final(self, best_epoch, best_acc, test_results=None):
        results = {
            'best_epoch': best_epoch,
            'best_val_acc': float(best_acc),
            'training_history': self.history
        }
        if test_results:
            results['test_results'] = test_results
            
        with open(os.path.join(self.log_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

# ==================== 数据集定义 ====================
class ForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        valid_folders = ['0_real', '1_fake']
        
        print(f"Scanning images in {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root)
            
            if folder_name in valid_folders:
                label = int(folder_name.split('_')[0])
                
                for img_name in files:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.samples.append(os.path.join(root, img_name))
                        self.labels.append(label)
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {root_dir}. Please check folder structure.")
        
        print(f"Total: {len(self.samples)} | Real(0): {self.labels.count(0)} | Fake(1): {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== 新增：MLP 定义 ====================
class DINOHeadConfig:
    """MLP的配置类"""
    def __init__(self, n_embd=1024, bias=True, dropout=0.1):
        self.n_embd = n_embd
        self.bias = bias
        self.dropout = dropout

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # n_embd: 输入维度 (对于 CLIP ViT-L 是 1024)
        # 4 * n_embd: 隐藏层维度 (4096)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) 
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)   # 1024 -> 4096
        x = self.gelu(x)   # 激活
        x = self.c_proj(x) # 4096 -> 1024
        x = self.dropout(x)# Dropout
        return x

# ==================== 模型定义 ====================
class DINOv2ForgeryDetectorLoRA(nn.Module):
    def __init__(self, lora_rank=8, lora_alpha=16):
        super().__init__()
        print("Loading DINOv2...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print(f"Injecting LoRA (Rank={lora_rank}, Alpha={lora_alpha})...")
        # 目标模块包括 qkv, proj 和 mlp
        self.target_modules=['qkv', 'proj', 'mlp']
        self.lora_params = inject_lora_to_linear(
            self.backbone, 
            rank=lora_rank, 
            alpha=lora_alpha,
            target_modules=self.target_modules
        )

        print(f"✅ 微调层: {self.target_modules}")
        
        self.feature_dim = 1024
        
        # ========== 修改部分开始 ==========
        # 定义配置
        config = DINOHeadConfig(n_embd=self.feature_dim, bias=True, dropout=0.1)
        
        # 使用 MLP + 线性层作为分类器
        # MLP将特征从 1024 映射回 1024 (增加了非线性能力)
        # 最后的 Linear 将 1024 映射到 1 (用于二分类 Logits)
        self.classifier = nn.Sequential(
            MLP(config),
            nn.Linear(self.feature_dim, 1)
        )
        # ========== 修改部分结束 ==========
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)
    
    def get_trainable_params(self):
        # 训练分类头参数 + LoRA参数
        return list(self.classifier.parameters()) + self.lora_params

# ==================== 训练与验证 ====================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, logger):
    model.train()
    total_loss = 0
    preds_list, labels_list = [], []
    
    pbar = tqdm(loader, desc="Training", leave=False)
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
    avg_loss = total_loss / len(loader)
    
    logger.info(f"Train Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")
    
    return avg_loss, acc

def evaluate(model, loader, criterion, device, logger, prefix='Val'):
    model.eval()
    total_loss = 0
    preds_list, labels_list, probs_list = [], [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=prefix, leave=False)
        for imgs, labels in pbar:
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
        'auc': roc_auc_score(labels_list, probs_list) if len(set(labels_list)) > 1 else 0.5,
        'precision': precision_recall_fscore_support(labels_list, preds_list, average='binary', zero_division=0)[0],
        'recall': precision_recall_fscore_support(labels_list, preds_list, average='binary', zero_division=0)[1],
        'f1': precision_recall_fscore_support(labels_list, preds_list, average='binary', zero_division=0)[2]
    }
    
    logger.info(f"{prefix} Loss: {metrics['loss']:.4f} | {prefix} Acc: {metrics['acc']:.4f}")
    logger.info(f"{prefix} AUC: {metrics['auc']:.4f} | {prefix} F1: {metrics['f1']:.4f}")
    
    return metrics

# ==================== 测试函数 ====================
def test_model(model_path, test_dir, log_dir, device):
    """在测试集上评估模型"""
    logger = logging.getLogger(__name__)
    
    # 加载模型 (结构必须一致)
    model = DINOv2ForgeryDetectorLoRA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {model_path}")
    
    # 准备测试数据
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = ForgeryDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 评估
    criterion = nn.BCEWithLogitsLoss()
    logger.info("\n" + "="*70)
    logger.info("Evaluating on Test Set...")
    logger.info("="*70)
    
    test_metrics = evaluate(model, test_loader, criterion, device, logger, prefix='Test')
    
    # 保存测试结果
    test_results = {
        'test_loss': float(test_metrics['loss']),
        'test_accuracy': float(test_metrics['acc']),
        'test_auc': float(test_metrics['auc']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'model_path': model_path,
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    logger.info(f"Test results saved to {log_dir}/test_results.json")
    logger.info("="*70)
    
    return test_results

# ==================== 主程序 ====================
def main():
    # 配置
    CFG = {
        'batch_size': 32,
        'lr': 5e-5,
        'epochs': 10,
        'lora_rank': 8,
        'lora_alpha': 16,
        'train_path': '/opt/data/private/data/train',
        'val_path': '/opt/data/private/data/val',
        'test_path': '/opt/data/private/data/test',
        'log_dir': './logs1',
        'ckpt_dir': './checkpoints'
    }
    
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    logger, log_file = setup_logger(CFG['log_dir'])
    train_logger = TrainingLogger(CFG['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*70)
    logger.info("DINOv2 LoRA Fine-tuning for Forgery Detection (MLP Head)")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(CFG, indent=2)}")
    
    # 数据增强
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
    
    # 数据集
    logger.info("\nLoading Datasets...")
    try:
        train_dataset = ForgeryDataset(CFG['train_path'], transform=train_transform)
        val_dataset = ForgeryDataset(CFG['val_path'], transform=val_transform)
    except Exception as e:
        logger.error(f"Dataset Error: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 模型
    model = DINOv2ForgeryDetectorLoRA(
        lora_rank=CFG['lora_rank'], 
        lora_alpha=CFG['lora_alpha']
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Ratio: {trainable_params/total_params*100:.2f}%")
    
    # 优化器与损失
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.get_trainable_params(), lr=CFG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = amp.GradScaler()
    
    # 训练循环
    logger.info("\n" + "="*70)
    logger.info("Starting Training...")
    logger.info("="*70)
    
    best_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(CFG['ckpt_dir'], 'best_model_alllora_transmlp.pth')
    
    for epoch in range(CFG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        logger.info("-"*70)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, logger
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device, logger, prefix='Val')
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录日志
        train_logger.log_epoch(epoch+1, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_auc': val_metrics['auc'],
            'lr': current_lr
        })
        
        # 保存最佳模型
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✓ Best model saved! (Acc: {best_acc:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CFG['ckpt_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc']
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info("\n" + "="*70)
    logger.info("Training Completed!")
    logger.info(f"Best Val Acc: {best_acc:.4f} at Epoch {best_epoch}")
    logger.info("="*70)
    
    # 在测试集上评估
    logger.info("\nEvaluating on Test Set...")
    test_results = test_model(best_model_path, CFG['test_path'], CFG['log_dir'], device)
    
    # 保存最终结果
    train_logger.save_final(best_epoch, best_acc, test_results)
    logger.info("All results saved.")

if __name__ == '__main__':
    main()
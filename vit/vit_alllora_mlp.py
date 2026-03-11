import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json
from datetime import datetime
import logging
import torch.cuda.amp as amp
from transformers import ViTModel

# ==================== 随机种子固定函数 ====================
def seed_everything(seed=42):
    """
    固定所有可能的随机种子以保证实验可复现性
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to: {seed}")

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

def inject_lora_to_linear(model, rank=8, alpha=16, target_modules=['query', 'key', 'value']):
    """
    为指定的 Linear 层注入 LoRA
    """
    lora_params = []
    print(f"Searching for modules with names containing: {target_modules}")
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        is_target = any(target in name for target in target_modules)
        
        if isinstance(module, nn.Linear) and is_target:
            print(f"Injecting LoRA to: {name}")
            in_features = module.in_features
            out_features = module.out_features
            
            # 冻结原始参数
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
            # 创建 LoRA 层
            lora_layer = LoRALayer(in_features, out_features, rank, alpha)
            
            # 替换 forward 方法
            original_forward = module.forward
            
            def make_lora_forward(orig_forward, lora):
                def forward(x):
                    return orig_forward(x) + lora(x)
                return forward
            
            module.forward = make_lora_forward(original_forward, lora_layer)
            
            # 挂载以便访问
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
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
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

# ==================== 模型定义 (ViT 修改版 - 包含 MLP 微调) ====================
class ViTForgeryDetectorLoRA(nn.Module):
    def __init__(self, lora_rank=8, lora_alpha=16):
        super().__init__()
        # 加载 google/vit-large-patch16-224
        print("Loading ViT Large Model (google/vit-large-patch16-224)...")
        self.backbone = ViTModel.from_pretrained("google/vit-large-patch16-224",add_pooling_layer=False)
        
        # 冻结所有主干参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print(f"Injecting LoRA (Rank={lora_rank}, Alpha={lora_alpha})...")
        print("Targets: QKV + MLP")
        
        # ==================== 核心修改点 ====================
        # target_modules 添加了 MLP 相关的层：
        # 'intermediate.dense': MLP 第一层
        # 'output.dense': MLP 第二层 (以及 Attention 输出层)
        self.lora_params = inject_lora_to_linear(
            self.backbone, 
            rank=lora_rank, 
            alpha=lora_alpha,
            target_modules=['query', 'key', 'value', 'intermediate.dense', 'output.dense'] 
        )
        # ==================================================
        
        self.feature_dim = self.backbone.config.hidden_size 
        
        # 分类器: Logits < 0 -> Real(0), Logits > 0 -> Fake(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        # ViT 输出包含 last_hidden_state 和 pooler_output
        outputs = self.backbone(pixel_values=x)
        
        # 使用 pooler_output (对应 [CLS] token 经过 MLP) 进行分类
        features = outputs.last_hidden_state[:, 0, :] 
        
        logits = self.classifier(features)
        return logits.squeeze(-1)
    
    def get_trainable_params(self):
        return list(self.classifier.parameters()) + self.lora_params

# ==================== 训练与验证 ====================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, logger):
    model.train()
    model.backbone.eval()
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
    
    # 实例化 ViT 模型
    model = ViTForgeryDetectorLoRA().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {model_path}")
    
    # ViT 标准归一化参数
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
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
        'seed': 42,
        'batch_size': 32,
        'lr': 5e-5,
        'epochs': 100,
        'lora_rank': 8,
        'lora_alpha': 16,
        'train_path': '/opt/data/private/data/train',
        'val_path': '/opt/data/private/data/val',
        'test_path': '/opt/data/private/data/test',
        'log_dir': './logs_vit',    
        'ckpt_dir': './checkpoints_vit' 
    }
    
    # 1. 设置随机种子
    seed_everything(CFG['seed'])
    
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    logger, log_file = setup_logger(CFG['log_dir'])
    train_logger = TrainingLogger(CFG['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*70)
    logger.info("ViT-Large LoRA (QKV + MLP) Fine-tuning for Forgery Detection")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(CFG, indent=2)}")
    
    # ViT 标准归一化参数 (Mean=0.5, Std=0.5)
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
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
    model = ViTForgeryDetectorLoRA(
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
    optimizer = optim.AdamW(model.get_trainable_params(), lr=CFG['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = amp.GradScaler()
    
    # 训练循环
    logger.info("\n" + "="*70)
    logger.info("Starting Training...")
    logger.info("="*70)
    
    best_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(CFG['ckpt_dir'], 'best_model_vit_alllora_mlp.pth')
    
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
        if (epoch + 1) % 5 == 0:
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
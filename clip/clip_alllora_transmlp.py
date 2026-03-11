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
import random

# 引入 Hugging Face 的 CLIP 模型
from transformers import CLIPVisionModel

# ==================== 0. 随机种子设置 ====================
def set_seed(seed=42):
    """固定所有随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")

# ==================== LoRA实现 (保持不变) ====================
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

def inject_lora_to_linear(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    lora_params = []
    for name, module in model.named_modules():
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

# ==================== 新增：自定义 MLP 模块 ====================
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

# ==================== 模型定义 (修改部分) ====================
class CLIPForgeryDetectorLoRA(nn.Module):
    def __init__(self, lora_rank=8, lora_alpha=16, mode='all'):
        super().__init__()
        print("Loading CLIP (ViT-L/14)...")
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # 冻结主干
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print(f"Injecting LoRA (Rank={lora_rank}, Alpha={lora_alpha})...")
        
        if mode == 'attention':
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        elif mode == 'all':
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']

        self.lora_params = inject_lora_to_linear(
            self.backbone, 
            rank=lora_rank, 
            alpha=lora_alpha,
            target_modules=self.target_modules
        )

        print(f"✅ 当前微调模式: {mode}")
        print(f"✅ 微调目标层: {self.target_modules}")

        # CLIP ViT-L/14 的 hidden_size
        self.feature_dim = 1024
        
        # ================== 修改：使用 MLP 分类头 ==================
        
        # 定义 MLP 的配置
        class MLPConfig:
            n_embd = 1024       # 输入维度
            bias = True         # 是否使用偏置
            dropout = 0.1       # Dropout 比率 (可调节)
            
        self.config = MLPConfig()
        
        # 1. 更加复杂的 MLP 投影层 (1024 -> 4096 -> 1024)
        self.mlp_head = MLP(self.config)
        
        # 2. 最终的二分类线性层 (1024 -> 1)
        # MLP 输出维度是 1024，需要映射到 1 维 Logit 进行 BCE Loss 计算
        self.final_classifier = nn.Linear(self.feature_dim, 1)
        
        # 初始化 final_classifier 的权重 (可选，有助于稳定初始训练)
        nn.init.normal_(self.final_classifier.weight, std=0.02)
        nn.init.zeros_(self.final_classifier.bias)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.pooler_output  # (batch, 1024)
        
        # 特征先经过 MLP
        x = self.mlp_head(features)
        
        # 再经过最终线性层
        logits = self.final_classifier(x)
        
        return logits.squeeze(-1)
    
    def get_trainable_params(self):
        # 需要同时训练：MLP 层参数 + 最终线性分类器参数 + LoRA 参数
        head_params = list(self.mlp_head.parameters()) + list(self.final_classifier.parameters())
        return head_params + self.lora_params

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
    logger = logging.getLogger(__name__)
    
    # 初始化模型结构
    model = CLIPForgeryDetectorLoRA(mode='all').to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {model_path}")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                           [0.26862954, 0.26130258, 0.27577711])
    ])
    
    test_dataset = ForgeryDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    logger.info("\n" + "="*70)
    logger.info("Evaluating on Test Set...")
    logger.info("="*70)
    
    test_metrics = evaluate(model, test_loader, criterion, device, logger, prefix='Test')
    
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
    CFG = {
        'seed': 42,
        'lora_mode':'all',
        'batch_size': 32,
        'lr': 5e-5,  # 由于MLP层数加深，可能需要适当调节LR，或者保持不变
        'epochs': 10,
        'lora_rank': 8,
        'lora_alpha': 16,
        'train_path': '/opt/data/private/data/train',
        'val_path': '/opt/data/private/data/val',
        'test_path': '/opt/data/private/data/test',
        'log_dir': './logs_clip', 
        'ckpt_dir': './checkpoints_clip'
    }
    
    set_seed(CFG['seed'])
    
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    logger, log_file = setup_logger(CFG['log_dir'])
    train_logger = TrainingLogger(CFG['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*70)
    logger.info("CLIP (ViT-L/14) LoRA + MLP Head for Forgery Detection")
    logger.info("="*70)
    logger.info(f"Config: {json.dumps(CFG, indent=2)}")
    
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])
    
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
    
    model = CLIPForgeryDetectorLoRA(
        lora_rank=CFG['lora_rank'], 
        lora_alpha=CFG['lora_alpha'],
        mode=CFG['lora_mode']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Ratio: {trainable_params/total_params*100:.2f}%")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.get_trainable_params(), lr=CFG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = amp.GradScaler()
    
    best_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(CFG['ckpt_dir'], 'best_model_clip_alllora_transmlp.pth')
    
    for epoch in range(CFG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        logger.info("-"*70)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, logger
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device, logger, prefix='Val')
        
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        train_logger.log_epoch(epoch+1, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_auc': val_metrics['auc'],
            'lr': current_lr
        })
        
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✓ Best model saved! (Acc: {best_acc:.4f})")
        
    logger.info("\n" + "="*70)
    logger.info(f"Best Val Acc: {best_acc:.4f} at Epoch {best_epoch}")
    
    logger.info("\nEvaluating on Test Set...")
    test_results = test_model(best_model_path, CFG['test_path'], CFG['log_dir'], device)
    train_logger.save_final(best_epoch, best_acc, test_results)

if __name__ == '__main__':
    main()
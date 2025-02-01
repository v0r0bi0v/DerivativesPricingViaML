import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import typing as tp
import glob


class Embedding(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        cnt_hidden_layers: int = 1, 
        hidden_layer_dims: tp.List[float] | None = None,
        normalize_output: bool = False
    ):
        super(Embedding, self).__init__()
        
        if hidden_layer_dims is None:
            hidden_layer_dims = [20] * cnt_hidden_layers
        
        assert len(hidden_layer_dims) == cnt_hidden_layers, \
            f"Length of hidden_layer_dims must be equal cnt_hidden_layers: {len(hidden_layer_dims)} != {cnt_hidden_layers}"
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, cnt_hidden_layers):
            layers.append(nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        self.normalize_output = normalize_output

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, state):
        output = self.model(state)
        if self.normalize_output:
            output = F.softmax(output, dim=-1)
        return output


def load_checkpoint(model_market, model_option_put, model_option_call, model_forward, optimizer, path):
    # Ищем все чекпоинты в директории
    checkpoint_files = sorted(glob.glob(f"{path}/*.pth"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not checkpoint_files:
        return 0, [], []  # Если чекпоинтов нет, возвращаем начальные значения
    
    # Загружаем последний чекпоинт
    last_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(last_checkpoint)

    # Загружаем состояние моделей и оптимизатора
    model_market.load_state_dict(checkpoint['model_market_state_dict'])
    model_option_put.load_state_dict(checkpoint['model_option_put_state_dict'])
    model_option_call.load_state_dict(checkpoint['model_option_call_state_dict'])
    model_forward.load_state_dict(checkpoint['model_forward_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Возвращаем последнюю эпоху, потери и другие данные
    return checkpoint['epoch'], checkpoint['losses_train'], checkpoint['losses_test']


def save_checkpoint(epoch, model_market, model_option_put, model_option_call, model_forward, optimizer, losses_train, losses_test, path):
    checkpoint = {
        'epoch': epoch,
        'model_market_state_dict': model_market.state_dict(),
        'model_option_put_state_dict': model_option_put.state_dict(),
        'model_option_call_state_dict': model_option_call.state_dict(),
        'model_forward_state_dict': model_forward.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses_train': losses_train,
        'losses_test': losses_test,
    }
    
    # Создаём имя файла с номером эпохи
    checkpoint_filename = f"{path}/checkpoint_epoch_{epoch}.pth"
    
    # Сохраняем чекпоинт
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_filename}")

# Advanced CIFAR10 MODEL

The code was produced by Claude Sonne,but I modified the parameters so that the mode could be used.
[Full code](summercampwork\day2\cifar_from_claude_sonnet4\cifar10_cnn_claudesonnet4.ipynb)
Best validation accuracy: 84.35%
# Key parameters(need modifying):

```python
class Config:
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    dropout_rate = 0.1
    use_mixed_precision = False
    use_advanced_augmentation = False
    early_stopping_patience = 5
    scheduler_type = 'cosine'
    ...
     # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
    ...
        self.fc = nn.Linear(256, 10)
    ...
```


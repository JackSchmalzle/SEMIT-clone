import os
from torchvision import datasets

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

base = 'datasets/cifar10'

for split in ['train', 'test']:
    dataset = datasets.CIFAR10('./data', train=(split=='train'), download=True)
    
    for c in CLASSES:
        os.makedirs(f'{base}/{split}_{c}', exist_ok=True)
    
    paths = []
    counts = {c: 0 for c in CLASSES}
    
    for img, label in dataset:
        c = CLASSES[label]
        paths.append(f'{split}_{c}/{counts[c]}.png')
        img.save(f'{base}/{paths[-1]}')
        counts[c] += 1
    
    with open(f'{base}/cifar10_list_{split}.txt', 'w') as f:
        f.write('\n'.join(paths))
    
    with open(f'{base}/cifar10_{split}_class_names.txt', 'w') as f:
        f.write('\n'.join([f'{split}_{c}' for c in CLASSES]))

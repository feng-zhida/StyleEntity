import torch
from tqdm import trange

import open_clip


@torch.no_grad()
def main():
    batch_size = 100
    file_name = './datasets/named_entity_for_train.txt'
    device = 'cuda'
    model, _ = clip.load('ViT-B/16', device)
    model.eval()

    with open(file_name) as f:
        named_entitys = f.read().strip().split('\n')
    features = torch.zeros(len(named_entitys), 768)
    for i in trange(len(named_entitys)//batch_size+1):
        text = named_entitys[i*batch_size:(i+1)*batch_size]
        if len(text) == 0:
            break
        text = tokenize(text).to(device)
        with torch.no_grad():
            out = model.encode_text(text)
            out = out.squeeze(0).cpu()
        features[i*batch_size:(i+1)*batch_size] = out
    torch.save(features, 'datasets/train_openai_clip_b.pt')


if __name__ == '__main__':
    main()
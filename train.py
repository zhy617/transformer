from data import *
from torch import nn
from torch.optim import Adam
import torch.optim as optim
from models.model.transformer import Transformer
from util.bleu import *
import time
from util.epoch_timer import epoch_time

def count_parameters(model: nn.Module):
    """
    Count the number of parameters with grad in the model
    Args:
        model: model to count the number of parameters
    Returns:
        int: number of parameters with grad in the model
    """
    # p.numel: return the number of elements in the tensor
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    """
    Initialize the weights of the model
    Args:
        m: model to initialize the weights
    """
    # If the model has weight and the dimension of the weight is greater than 1
    # except for ReLU, Bias
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    d_model=d_model,
                    d_ff=ffn_hidden,
                    n_heads=n_heads,
                    max_len=max_len,
                    n_layers=n_layers,
                    drop_prob=dropout,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

optimizer = Adam(model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    """
    train the model
    Args:
        model: model to train
        iterator: iterator for the dataset
        optimizer: optimizer for the model
        criterion: loss function
        clip: gradient clipping value
    Returns:
        float: average loss
    """
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        
        # remove the last token of the target 'EOS'
        # output: [batch_size, trg_len - 1, trg_vocab_size]
        output = model(src, trg[:, :-1])

        # output_reshape: [batch_size * (trg_len - 1), trg_vocab_size]
        output_reshape = output.contiguous().view(-1, output.shape[-1])

        # trg shape: [batch_size, trg_len]
        # trg_reshape: [batch_size * (trg_len - 1)]
        # remove the first token of the target 'SOS'
        trg_reshape = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg_reshape)
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        print(f'Batch: {i+1:02} | Loss: {loss.item():.3f}')
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """
    evaluate the model
    Args:
        model: model to evaluate
        iterator: iterator for the dataset
        criterion: loss function
    Returns:
        (float, float): average loss, average BLEU score
    """
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output:torch.Tensor = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            # calculate the BLEU score
            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], data_loader.target.vocab)
                    output_words = idx_to_word(output[j].argmax(dim=1), data_loader.target.vocab)
                    bleu = get_bleu(output_words.split(), trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
            total_bleu = np.mean(total_bleu)
            batch_bleu.append(total_bleu)

    return epoch_loss / len(iterator), np.mean(batch_bleu)

def run(total_epoch, best_loss):
    train_losses, valid_losses, valid_bleus = [], [], []
    for epoch in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # after warmup, adjust the learning rate
        # based on the validation loss
        # if the validation loss does not decrease for 'patience' epochs,
        # reduce the learning rate by 'factor'
        if epoch > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_bleus.append(bleu)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model{0}.pt'.format(valid_loss))
        
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/valid_loss.txt', 'w')
        f.write(str(valid_losses))
        f.close()

        f = open('result/valid_bleu.txt', 'w')
        f.write(str(valid_bleus))
        f.close()

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {np.exp(valid_loss):7.3f}')
        print(f'\t Val. BLEU: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch = epoch, best_loss = inf)
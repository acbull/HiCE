# coding: utf-8
import argparse
from model import *
from util import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm   # Comment this line if using jupyter notebook


parser = argparse.ArgumentParser(description='Training HiCE on WikiText-103')

'''
Dataset arguments
'''
parser.add_argument('--w2v_dir', type=str, default='./data/base_w2v/wiki_all.sent.split.model',
                    help='location of the default node embedding')
parser.add_argument('--corpus_dir', type=str, default='./data/wikitext-103/',
                    help='location of the training corpus (wikitext-103)')
parser.add_argument('--freq_lbound', type=int, default=16,
                    help='Lower bound of word frequency in w2v for selecting target words')
parser.add_argument('--freq_ubound', type=int, default=2 ** 16,
                    help='Upper bound of word frequency in w2v for selecting target words')
parser.add_argument('--cxt_lbound', type=int, default=2,
                    help='Lower bound of word frequency in corpus for selecting target words')
parser.add_argument('--chimera_dir', type=str, default='./data/chimeras/',
                    help='location of the testing corpus (Chimeras)')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')
'''
Model hyperparameters
'''
parser.add_argument('--maxlen', type=int, default=12,
                    help='maxlen of context (half, left or right) and character')
parser.add_argument('--use_morph', action='store_true',
                    help='initial learning rate')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of hidden units per layer')
parser.add_argument('--n_layer', type=int, default=2,
                    help='number of encoding layers')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='upper bound of training epochs')
parser.add_argument('--n_batch', type=int, default=256, 
                    help='batch size')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--lr_init', type=float, default=1e-3,
                    help='initial learning rate for Adam')
parser.add_argument('--n_shot', type=int, default=10,
                    help='upper bound of training K-shot')
'''
Validation & Test arguments
'''
parser.add_argument('--test_interval', type=int, default=1,
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='./save/',
                    help='location for saving the best model')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='Learning Rate Decay using ReduceLROnPlateau Scheduler')
parser.add_argument('--threshold', type=float, default=1e-3,
                    help='Learning Rate Decay using ReduceLROnPlateau Scheduler')
parser.add_argument('--patience', type=int, default=4,
                    help='Patience for lr Scheduler judgement')
parser.add_argument('--lr_early_stop', type=float, default=1e-5,
                    help='the lower bound of training lr. Early stop after lr is below it.')


'''
Adaptation with First-Order MAML arguments
'''
parser.add_argument('--adapt', action='store_true',
                    help='adapt to target dataset with 1-st order MAML')
parser.add_argument('--inner_batch_size', type=int, default=4,
                    help='batch for updating using source corpus')
parser.add_argument('--meta_batch_size', type=int, default=16,
                    help='batch for accumulating meta gradients')

args = parser.parse_args()

def get_batch(words, dataset, w2v, batch_size, k_shot, device):
    sample_words = np.random.choice(words, batch_size)
    contexts = []
    targets  = []
    vocabs   = []
    for word in sample_words:
        if len(dataset[word]) != 0:
            sample_sent_idx = np.random.choice(len(dataset[word]), k_shot)
            sample_sents = dataset[word][sample_sent_idx]
            contexts += [sample_sents]
            targets  += [w2v.wv[word]]
            vocabs   += [[_vocab[vi] for vi in word if vi in _vocab]]
    contexts = torch.LongTensor(contexts).to(device)
    targets  = Variable(torch.FloatTensor(targets).to(device))
    vocabs   = torch.LongTensor(pad_sequences(vocabs, maxlen = args.maxlen * 2)).to(device)
    return contexts, targets, vocabs

def evaluate_on_chimera(model, chimera_data):
    model.eval()
    with torch.no_grad():
        for k_shot in chimera_data:
            data = chimera_data[k_shot]
            test_contexts = torch.LongTensor(data['contexts']).to(device)
            test_targets  = torch.FloatTensor(data['ground_truth_vector']).to(device)
            test_vocabs   = torch.LongTensor(data['character']).to(device)
            test_pred     = model.forward(test_contexts, test_vocabs)
            cosine = F.cosine_similarity(test_pred, test_targets).mean().cpu().tolist()

            test_prb = np.array(list(data["probes"]))
            test_scr = np.array(list(data["scores"]))
            cors = []
            prov = [[base_w2v.wv[pi] for pi in probe] for probe in test_prb]
            for p1, p2, p3 in zip(test_pred.cpu().numpy(), prov, test_scr):
                cos = cosine_similarity([p1], p2)
                cor = spearmanr(cos[0], p3)[0]
                cors += [cor]
            print('-' * 100)
            print("Test with %d shot: Cosine: %.4f;  Spearman: %.4f" % (k_shot, cosine, np.average(cors)))    


_vocab = {v: i+1 for v, i in zip('abcdefghijklmnopqrstuvwxyz', range(26))}
base_w2v = Word2Vec.load(args.w2v_dir)
source_train_dataset, source_valid_dataset, dictionary = load_training_corpus(base_w2v, args.corpus_dir, \
     maxlen = args.maxlen, freq_lbound = args.freq_lbound, freq_ubound = args.freq_ubound, cxt_lbound = args.cxt_lbound)
chimera_data = load_chimera(dictionary = dictionary, base_w2v = base_w2v, chimera_dir = args.chimera_dir)



device = torch.device("cuda:%d" % args.cuda if args.cuda != -1 else "cpu")
model = HICE(n_head = args.n_head,  n_hid = base_w2v.vector_size, n_seq = args.maxlen * 2, \
            n_layer= args.n_layer, w2v = dictionary.idx2vec, use_morph=args.use_morph).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_init)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience = args.patience, mode='max', threshold=args.threshold)


source_train_words = list(source_train_dataset.keys())
source_valid_words = list(source_valid_dataset.keys())


loss_stat = []
best_valid_cosine = -1
for epoch in np.arange(args.n_epochs) + 1:
    print('=' * 100)
    train_cosine = []
    valid_cosine = []
    model.train()
    with tqdm(np.arange(args.n_batch), desc='Train') as monitor:
        for batch in monitor:
            k_shot = np.random.randint(args.n_shot) + 1 # randomly sample a context length, and only give the model with this size of contexts
            train_contexts, train_targets, train_vocabs = get_batch(words = source_train_words, dataset = source_train_dataset, \
                                       w2v = base_w2v, batch_size = args.batch_size, k_shot = k_shot, device = device)
            optimizer.zero_grad()
            pred_emb = model.forward(train_contexts, train_vocabs)
            loss = -F.cosine_similarity(pred_emb, train_targets).mean()
            loss.backward()
            optimizer.step()
            train_cosine += [[-loss.cpu().detach().numpy(), k_shot]]
            monitor.set_postfix(train_status = train_cosine[-1])
    model.eval()
    with torch.no_grad():
        with tqdm(np.arange(args.n_batch // args.n_shot), desc='Valid') as monitor:
            for batch in monitor:
                for k_shot in np.arange(args.n_shot) + 1: # during evaluation, use all the possible context length
                    valid_contexts, valid_targets, valid_vocabs = get_batch(words = source_valid_words, dataset = source_valid_dataset, \
                                       w2v = base_w2v, batch_size = args.batch_size, k_shot = k_shot, device = device)
                    pred_emb = model.forward(valid_contexts, valid_vocabs)
                    loss = -F.cosine_similarity(pred_emb, valid_targets).mean()
                    valid_cosine += [[-loss.cpu().numpy(), k_shot]]
                    monitor.set_postfix(valid_status = valid_cosine[-1])
    print('-' * 100)
    avg_train, avg_valid = np.average(train_cosine, axis=0)[0], np.average(valid_cosine, axis=0)[0]
    print(("Epoch: %d: Train Cosine: %.4f; Valid Cosine: %.4f; LR: %f") \
            % (epoch, avg_train, avg_valid, optimizer.param_groups[0]['lr']))
    scheduler.step(avg_valid)
    
    if avg_valid > best_valid_cosine:
        best_valid_cosine = avg_valid
        with open(os.path.join(args.save_dir, 'model.pt'), 'wb') as f:
            torch.save(model, f)
        with open(os.path.join(args.save_dir, 'optimizer.pt'), 'wb') as f:
            torch.save(optimizer.state_dict(), f)
    
    loss_stat += [[epoch, li, 'TRAIN', k_shot] for (li, k_shot) in train_cosine]
    loss_stat += [[epoch, li, 'VALID', k_shot] for (li, k_shot) in valid_cosine]
    if epoch % args.test_interval == 0:
        '''
        # This script can plot loss curve and position attention weight for debugging.
        plot_stat = pd.DataFrame(loss_stat, columns=['Epoch', 'Cosine', 'Data', 'K-shot'])
        print(model.bal)
        print(model.pos_att.pos_att)
        for k_shot in [2,4,6]:
            data = plot_stat[plot_stat['K-shot'] == k]
            sb.lineplot(x='Epoch', y='Cosine', hue='Data', data = data)
            plt.title('K-shot = ' + str(k))
            plt.savefig(args.save_dir + 'training_curve_%d.png' % k)
        '''
        evaluate_on_chimera(model, chimera_data)
    if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
        print('Finish Training')
        break
'''
Evaluate on the best model:
'''
model = torch.load(os.path.join(args.save_dir, 'model.pt')).to(device)
print('=' * 100)
print('Evaluate on the best model with supervised training on source corpus:')
evaluate_on_chimera(model, chimera_data)

    
if args.adapt:
    best_score = -1
    target_train_dataset, target_valid_dataset, target_dictionary = load_training_corpus(base_w2v, args.chimera_dir, maxlen = args.maxlen,\
         freq_lbound = args.freq_lbound, freq_ubound = args.freq_ubound, cxt_lbound = args.cxt_lbound, dictionary = dictionary)
    target_train_words = list(target_train_dataset.keys())
    target_valid_words = list(target_valid_dataset.keys())
    model.update_embedding(target_dictionary.idx2vec)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_init * args.lr_decay)
    optimizer.zero_grad() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience = args.patience, mode='max', threshold=args.threshold)
    '''
    Use a temp model to calculate update on source task, then calculate the gradient with updated weights on target task. 
    Finally pass the gradient to original model and conduct optimization (gradient descent)
    '''
    model_tmp = copy.deepcopy(model)
    for meta_epoch in np.arange(args.n_epochs):
        print('=' * 100)
        source_cosine = []
        target_cosine = []
        meta_grads = []
        with tqdm(np.arange(args.meta_batch_size), desc='Meta Train') as monitor:
            for meta_batch in monitor:
                model_tmp.load_state_dict(model.state_dict())
                model_tmp.train()
                optimizer_tmp = torch.optim.Adam(model_tmp.parameters(), lr = 5e-4)
                '''
                Cumulate Inner Gradient
                '''
                for inner_batch in range(args.inner_batch_size):
                    k_shot = np.random.randint(args.n_shot) + 1
                    source_train_contexts, source_train_targets, source_train_vocabs = get_batch(words = source_train_words, \
                        dataset = source_train_dataset, w2v = base_w2v, batch_size = args.batch_size, k_shot = k_shot, device = device)
                    optimizer_tmp.zero_grad()
                    pred_emb = model_tmp.forward(source_train_contexts, source_train_vocabs)
                    loss = -F.cosine_similarity(pred_emb, source_train_targets).mean()
                    loss.backward()
                    optimizer_tmp.step()
                model_tmp.eval()
                optimizer_tmp.zero_grad()
                k_shot = np.random.randint(args.n_shot) + 1
                target_train_contexts, target_train_targets, target_train_vocabs = get_batch(words = target_train_words, \
                    dataset = target_train_dataset, w2v = base_w2v, batch_size = args.batch_size, k_shot = k_shot, device = device)
                pred_emb = model_tmp.forward(target_train_contexts, target_train_vocabs)
                loss = -F.cosine_similarity(pred_emb, target_train_targets).mean()
                loss.backward()
                meta_grads += [{name: param.grad for (name, param) in model_tmp.named_parameters() if param.requires_grad}]
            # end for meta_batch in tqdm:
        # end with tqdm(np.arange(args.meta_batch_size), desc='Meta Train') as monitor:
        '''
        Meta-Update
        '''
        meta_grads = {name: torch.stack([name_grad[name] for name_grad in meta_grads]).mean(dim=0)
                                      for name in meta_grads[0].keys()}
        hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                hooks.append(
                    param.register_hook(replace_grad(meta_grads, name))
                )

        model.train()
        optimizer.zero_grad() 
        pred_emb = model.forward(target_train_contexts, target_train_vocabs) 
        # Here the data (forwad, loss) doesn't matter at all, as the gradient will be replaced when "loss.backward()" with meta_grads
        loss = -F.cosine_similarity(pred_emb, target_train_targets).mean()
        loss.backward()
        optimizer.step()

        for h in hooks:
            h.remove()
            
        '''
        Validate using either of the updated model_tmp (we use the last one for convenience)
        '''
        model_tmp.eval()
        with tqdm(np.arange(args.n_batch // args.n_shot), desc='Meta Valid') as monitor:
            for batch in monitor:
                for k_shot in np.arange(args.n_shot) + 1:
                    source_valid_contexts, source_valid_targets, source_valid_vocabs = get_batch(words = source_valid_words, \
                    dataset = source_valid_dataset, w2v = base_w2v, batch_size = args.batch_size, k_shot = k_shot, device = device)
                    pred_emb = model_tmp.forward(source_valid_contexts, source_valid_vocabs)
                    source_cosine += [F.cosine_similarity(pred_emb, source_valid_targets).mean().cpu().detach().numpy()]

                    target_valid_contexts, target_valid_targets, target_valid_vocabs = get_batch(words = target_valid_words, \
                        dataset = target_valid_dataset, w2v = base_w2v, batch_size = args.batch_size, k_shot = k_shot, device = device)
                    pred_emb = model_tmp.forward(target_valid_contexts, target_valid_vocabs)
                    target_cosine += [F.cosine_similarity(pred_emb, target_valid_targets).mean().cpu().detach().numpy()]
        print('-' * 100)
        avg_train, avg_valid = np.average(source_cosine), np.average(target_cosine)
        print(("Epoch: %d: Meta Train Cosine: %.4f; Meta Valid Cosine: %.4f; LR: %f") \
            % (meta_epoch, avg_train, avg_valid, optimizer.param_groups[0]['lr']))
        score = avg_train + avg_valid
        scheduler.step(score)
        if score > best_score:
            best_score = score
            with open(os.path.join(args.save_dir, 'meta_model.pt'), 'wb') as f:
                torch.save(model_tmp, f)
            with open(os.path.join(args.save_dir, 'meta_optimizer.pt'), 'wb') as f:
                torch.save(optimizer.state_dict(), f)
        evaluate_on_chimera(model_tmp, chimera_data)
        # end with torch.no_grad():
        print('-' * 100)
        if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            print('Finish Training')
            break
    # end for meta_epoch in np.arange(args.n_epochs):
# end if args.adapt:


model = torch.load(os.path.join(args.save_dir, 'meta_model.pt')).to(device)
print('=' * 100)
print('Evaluate on the best model with meta training on both source and target corpus:')
evaluate_on_chimera(model, chimera_data)
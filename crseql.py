import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

def _cosine_similarity(a, b):
    a_norm = torch.linalg.norm(a)
    b_norm = torch.linalg.norm(b)
    a_b_dot = torch.inner(a, b)
    return torch.mean(a_b_dot / (a_norm * b_norm))

def _l1_norm(a, b):
    add_inv_b = torch.mul(b, -1)
    summation = torch.add(a, add_inv_b)
    abs_val = torch.abs(summation)
    return torch.sum(abs_val)

def _l2_norm(a, b):
    add_inv_b = torch.mul(b, -1)
    summation = torch.add(a, add_inv_b)
    square = torch.mul(summation, summation)
    sqrt = torch.sqrt(square)
    return torch.sum(sqrt)

def cross_sequence_learning():
    testlabel = []
    testpred = []
    testloss = []
    testacc = []
    testprob = []


    for i, (train_index, val_index) in enumerate(skf.split(traindata1, trainidx)):
        
        
        print('[Fold %d/%d]' % (i + 1, kfold))
        
        X_train, X_valid = traindata1[train_index], traindata1[val_index]
        y_train, y_valid = trainidx[train_index], trainidx[val_index]
        
        # Split the original and ROI-cropped images with respect to their classes - the given classes of input pairs aren't mixed up.
        
        original_0 = []
        original_1 = []
        
        for j in range(len(y_train)):
            if y_train[j] == 0:
                original_0.append(traindata1[train_index[j]])
            else:
                original_1.append(traindata1[train_index[j]])
        
        cropped_0 = []
        cropped_1 = []
        
        for k in range(len(y_train)):
            if y_train[k] == 0:
                cropped_0.append(traindata2[train_index[k]])
            else:
                cropped_1.append(traindata2[train_index[k]])
        
        # Calculate cosine similarities

        metric_0 = []
        metric_1 = []
        
        for a in range(len(original_0)):
            for b in range(len(cropped_0)):
                metric_0.append((_cosine_similarity(original_0[a][0], cropped_0[b][0])).detach().cpu().numpy().item())

        for c in range(len(original_1)):
            for d in range(len(cropped_1)):
                metric_1.append((_cosine_similarity(original_1[c][0], cropped_1[d][0])).detach().cpu().numpy().item())
        
        # List chunk (List comprehension)
        
        metric_0_chunk = [metric_0[e * len(cropped_0):(e + 1) * len(cropped_0)] for e in range((len(metric_0) + len(cropped_0) - 1) // len(cropped_0) )]
        metric_1_chunk = [metric_1[f * len(cropped_1):(f + 1) * len(cropped_1)] for f in range((len(metric_1) + len(cropped_1) - 1) // len(cropped_1) )]
        
        # Find the index having the lowest cosine similarity
        
        min_0 = []
        min_1 = []
        
        for g in range(len(metric_0_chunk)):
            min_0.append(np.argmin(metric_0_chunk[g]))
            
            # Removing duplicates process
            for n in range(len(metric_0_chunk)):
                metric_0_chunk[n][min_0[g]] = np.inf        

        for h in range(len(metric_1_chunk)):
            min_1.append(np.argmin(metric_1_chunk[h]))
            
            # Removing duplicates process
            for o in range(len(metric_1_chunk)):
                metric_1_chunk[o][min_1[h]] = np.inf  
        
        # Aggregate two matched images

        cropped_0_sorted = []
        cropped_1_sorted = []
        
        for l in range(len(min_0)):
            cropped_0_sorted.append(cropped_0[min_0[l]])

        for m in range(len(min_1)):
            cropped_1_sorted.append(cropped_1[min_1[m]])

        new_traindata2 = []
        
        cropped_num0 = 0
        cropped_num1 = 0
        
        for k in range(len(y_train)):
            if y_train[k] == 0:
                new_traindata2.append(cropped_0_sorted[cropped_num0])
                cropped_num0 += 1

            else:
                new_traindata2.append(cropped_1_sorted[cropped_num1])
                cropped_num1 += 1
        
        new_traindata2 = np.array(new_traindata2)
        
        # There is no need to implement Cross-Sequence Learning for validation and test process. 
        X_train2, X_valid2 = new_traindata2, traindata2[val_index]
        
        
        trainfinal = []

        for h in range(X_train.shape[0]):
        trainfinal.append((X_train[h], X_train2[h], y_train[h]))

        valfinal = []

        for t in range(X_valid.shape[0]):
        valfinal.append((X_valid[t], X_valid2[t], y_valid[t]))

        trainloader = DataLoader(trainfinal, batch_size = batchsize, shuffle = False)
        validloader = DataLoader(valfinal, batch_size = batchsize, shuffle = False)
        
        print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
        


        # Train Process
        epochval = []
        valloss = []
        valacc = []

        for epoch in range(1, epochs + 1):
            train(model_merged, trainloader, optimizer, log_interval = 5)
            valid_loss, valid_accuracy, _, _, _ = evaluate(model_merged, validloader)
            epochval.append(epoch)
            valloss.append(valid_loss)
            valacc.append(valid_accuracy)
            print("\n[EPOCH: {}], \tValidation Loss: {:.6f}, \tValidation Accuracy: {:.6f} % \n".format(
                epoch, valid_loss, valid_accuracy))
        

        # Validation and Test Process
        test_loss, test_accuracy, label, pred, prob = evaluate(model_merged, testloader)
        testlabel.append(label)
        testpred.append(pred)
        testloss.append(test_loss)
        testacc.append(test_accuracy)
        testprob.append(prob)

        print("\nTest Loss: {:.4f}, \tTest Accuracy: {:.4f} % \n".format(test_loss, test_accuracy))
# -*- coding: utf-8 -*-
import copy
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from timeit import default_timer as timer

from sklearn.model_selection import cross_val_score

from .layers import BioEmbedding, CKNLayer, POOLINGS, LinearMax, PREPROCESSORS

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class CKNSequential(nn.Module):
    def __init__(self, in_channels, out_channels_list, filter_sizes,
                 subsamplings, kernel_funcs=None, kernel_args_list=None,
                 kernel_args_trainable=False, **kwargs):
        assert len(out_channels_list) == len(filter_sizes) == len(subsamplings), "incompatible dimensions"

        super(CKNSequential, self).__init__()

        self.n_layers = len(out_channels_list)
        self.in_channels = in_channels
        self.out_channels = out_channels_list[-1]
        self.filter_sizes = filter_sizes
        self.subsamplings = subsamplings

        ckn_layers = []

        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.3
            else:
                kernel_args = kernel_args_list[i]

            ckn_layer = CKNLayer(in_channels, out_channels_list[i],
                                 filter_sizes[i], subsampling=subsamplings[i],
                                 kernel_func=kernel_func,
                                 kernel_args=kernel_args,
                                 kernel_args_trainable=kernel_args_trainable,
                                 **kwargs)

            ckn_layers.append(ckn_layer)
            in_channels = out_channels_list[i]

        self.ckn_layers = nn.Sequential(*ckn_layers)

    def __getitem__(self, idx):
        return self.ckn_layers[idx]

    def __len__(self):
        return len(self.ckn_layers)

    def __iter__(self):
        return iter(self.ckn_layers._modules.values())

    def forward_at(self, x, i=0):
        assert x.size(1) == self.ckn_layers[i].in_channels, "bad dimension"
        return self.ckn_layers[i](x)

    def forward(self, x):
        return self.ckn_layers(x)

    def representation(self, x, n=0):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            x = self.forward_at(x, i)
        return x

    def compute_mask(self, mask=None, n=-1):
        if mask is None:
            return mask
        if n > self.n_layers:
            raise ValueError("Index larger than number of layers")
        if n == -1:
            n = self.n_layers
        for i in range(n):
            mask = self.ckn_layers[i].compute_mask(mask)
        return mask

    def normalize_(self):
        for module in self.ckn_layers:
            module.normalize_()

    @property
    def len_motif(self):
        l = self.filter_sizes[self.n_layers - 1]
        for i in reversed(range(1, self.n_layers)):
            l = self.subsamplings[i - 1] * l + self.filter_sizes[i - 1] - 2
        return l


class CKN(nn.Module):
    def __init__(self, in_channels, out_channels_list, filter_sizes,
                 subsamplings, kernel_funcs=None, kernel_args_list=None,
                 kernel_args_trainable=False, alpha=0., fit_bias=True,
                 reverse_complement=False, global_pool='mean',
                 penalty='l2', scaler='standard_row', no_embed=False,
                 encoding='one_hot', global_pool_arg=1e-03, n_class=1, mask_zeros=True, **kwargs):
        super(CKN, self).__init__()
        self.reverse_complement = reverse_complement
        self.embed_layer = BioEmbedding(
            in_channels, reverse_complement, mask_zeros=mask_zeros, no_embed=no_embed, encoding=encoding)
        self.ckn_model = CKNSequential(
            in_channels, out_channels_list, filter_sizes,
            subsamplings, kernel_funcs, kernel_args_list,
            kernel_args_trainable, **kwargs)
        self.global_pool = POOLINGS[global_pool]()
        self.global_pool.alpha = global_pool_arg
        self.out_features = out_channels_list[-1]
        self.n_class = n_class
        self.initialize_scaler(scaler)
        self.classifier = LinearMax(self.out_features, n_class, alpha=alpha,
                                    fit_bias=fit_bias,
                                    reverse_complement=reverse_complement,
                                    penalty=penalty)

    def initialize_scaler(self, scaler=None):
        pass

    def normalize_(self):
        self.ckn_model.normalize_()

    def representation_at(self, input, n=0):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model.representation(output, n)
        mask = self.ckn_model.compute_mask(mask, n)
        return output, mask

    def representation(self, input):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model(output)
        mask = self.ckn_model.compute_mask(mask)
        output = self.global_pool(output, mask)
        return output

    def forward(self, input, proba=False):
        output = self.representation(input)
        return self.classifier(output, proba)

    def unsup_train_ckn(self, data_loader, n_sampling_patches=100000,
                        init=None, use_cuda=False, n_patches_per_batch=None):
        self.train(False)
        if use_cuda:
            self.cuda()
        for i, ckn_layer in enumerate(self.ckn_model):
            print("Training layer {}".format(i))
            n_patches = 0
            if n_patches_per_batch is None:
                try:
                    n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader)
                except:
                    n_patches_per_batch = 1000
            patches = torch.Tensor(n_sampling_patches, ckn_layer.patch_dim)
            if use_cuda:
                patches = patches.cuda()

            for data, _ in data_loader:
                if n_patches >= n_sampling_patches:
                    break
                if use_cuda:
                    data = data.cuda()
                with torch.no_grad():
                    data, mask = self.representation_at(data, i)
                    data_patches = ckn_layer.sample_patches(
                        data, mask, n_patches_per_batch)
                size = data_patches.size(0)
                if n_patches + size > n_sampling_patches:
                    size = n_sampling_patches - n_patches
                    data_patches = data_patches[:size]
                patches[n_patches: n_patches + size] = data_patches
                n_patches += size

            print("total number of patches: {}".format(n_patches))
            patches = patches[:n_patches]
            ckn_layer.unsup_train(patches, init=init)

    def unsup_train_classifier(self, data_loader, criterion=None,
                               use_cuda=False):
        encoded_train, encoded_target = self.predict(
            data_loader, True, use_cuda=use_cuda)
        print(encoded_train.shape)
        if hasattr(self, 'scaler') and not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            encoded_train = self.scaler.fit_transform(
                encoded_train.view(-1, self.out_features)
            ).view(size, -1)
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False,
                proba=False, use_cuda=False):
        self.train(False)
        if use_cuda:
            self.cuda()
        n_samples = len(data_loader.dataset)
        # if self.n_class == 1:
        #     target_output = torch.Tensor(n_samples)
        # else:
        #     target_output = torch.LongTensor(n_samples)
        batch_start = 0
        for i, (data, target, *_) in enumerate(data_loader):
            batch_size = data.shape[0]
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data, proba).data.cpu()
            if self.reverse_complement:
                batch_out = torch.cat(
                    (batch_out[:batch_size], batch_out[batch_size:]), dim=-1)
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = target.new_empty([n_samples]+list(target.shape[1:]))
            output[batch_start:batch_start+batch_size] = batch_out
            target_output[batch_start:batch_start+batch_size] = target
            batch_start += batch_size
        output.squeeze_(-1)
        return output, target_output

    def compute_motif(self, max_iter=2000):
        self.train(True)
        weights = self.classifier.weight.data.cpu().clone().numpy()
        weights = weights.ravel()
        indices = np.argsort(np.abs(weights))[::-1]
        pwm_all = []
        for index in indices:
            motif, loss = optimize_motif(index, self.ckn_model, max_iter)
            motif_norm = np.linalg.norm(motif)
            threshold = (1 - motif_norm * np.exp(-4.5)) ** 2
            if loss < threshold:
                print("filter {} is good".format(index))
                pwm_all.append(motif)

        pwm_all = np.asarray(pwm_all)
        return pwm_all


class unsupCKN(CKN):
    def initialize_scaler(self, scaler=None):
        self.scaler = PREPROCESSORS[scaler]()

    def unsup_train(self, data_loader, n_sampling_patches=500000,
                    use_cuda=False):
        self.train(False)
        print("Training CKN layers")
        tic = timer()
        self.unsup_train_ckn(data_loader, n_sampling_patches, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))
        print("Training classifier")
        tic = timer()
        self.unsup_train_classifier(data_loader, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

    def unsup_cross_val(self, data_loader, pos_data_loader=None,
                        n_sampling_patches=500000,
                        alpha_grid=None, kfold=5,
                        scoring='neg_log_loss',
                        init_kmeans=None, balanced=False, use_cuda=False):
        self.train(False)
        if alpha_grid is None:
            alpha_grid = [1.0, 0.1, 0.01, 0.001]
        print("Training CKN layers")
        tic = timer()
        if pos_data_loader is not None:
            self.unsup_train_ckn(pos_data_loader, n_sampling_patches,
                init=init_kmeans, use_cuda=use_cuda)
        else:
            self.unsup_train_ckn(data_loader, n_sampling_patches,
                init=init_kmeans, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

        print("Start cross-validation")
        best_score = -float('inf')
        best_alpha = 0
        tic = timer()
        encoded_train, encoded_target = self.predict(
            data_loader, True, use_cuda=use_cuda)
        if not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            encoded_train = self.scaler.fit_transform(
                encoded_train.view(-1, self.out_features)
            ).view(size, -1)
        # self.cpu()
        if not balanced:
            clf = self.classifier
            if use_cuda:
                n_jobs = None
            else:
                n_jobs = -1
            for alpha in alpha_grid:
                print("lambda={}".format(alpha))
                clf.alpha = alpha
                clf.reset_parameters()
                score = cross_val_score(clf, encoded_train.numpy(),
                                        encoded_target.numpy(),
                                        cv=kfold, scoring=scoring, n_jobs=n_jobs)
                score = score.mean()
                print("val score={}".format(score))
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            print("best lambda={}".format(best_alpha))
            clf.alpha = best_alpha
            clf.fit(encoded_train, encoded_target)
            toc = timer()

        else:
            for alpha in alpha_grid:
                print("lambda={}".format(alpha))
                #clf = LinearSVC(C=1./alpha, fit_intercept=False, class_weight='balanced')
                clf = LogisticRegression(C=1./alpha, fit_intercept=False, class_weight='balanced', solver='liblinear')
                score = cross_val_score(clf, encoded_train.numpy(),
                                        encoded_target.numpy(),
                                        cv=kfold, scoring=scoring, n_jobs=-1)
                score = score.mean()
                print("val score={}".format(score))
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            print("best lambda={}".format(best_alpha))
            #clf = LinearSVC(C=1./best_alpha, fit_intercept=False, class_weight='balanced')
            clf = LogisticRegression(C=1./best_alpha, fit_intercept=False, class_weight='balanced', solver='liblinear')
            clf.fit(encoded_train.numpy(), encoded_target.numpy())
            toc = timer()
            self.classifier.weight.data.copy_(torch.from_numpy(clf.coef_.reshape(1, -1)))
            print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

    def representation(self, input):
        output = super(unsupCKN, self).representation(input)
        return self.scaler(output)


class supCKN(CKN):
    def sup_train(self, train_loader, criterion, optimizer, lr_scheduler=None,
                  init_train_loader=None, epochs=100, val_loader=None,
                  n_sampling_patches=500000, unsup_init=None,
                  use_cuda=False, early_stop=True):
        print("Initializing CKN layers")
        tic = timer()
        if init_train_loader is not None:
            self.unsup_train_ckn(init_train_loader, n_sampling_patches,
                                 init=unsup_init, use_cuda=use_cuda)
        else:
            self.unsup_train_ckn(train_loader, n_sampling_patches,
                                 init=unsup_init, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

        phases = ['train']
        data_loader = {'train': train_loader}
        if val_loader is not None:
            phases.append('val')
            data_loader['val'] = val_loader

        epoch_loss = None
        best_loss = float('inf')
        best_acc = 0
        best_epoch = 0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)
            self.train(False)
            self.unsup_train_classifier(
                data_loader['train'], criterion, use_cuda=use_cuda)

            for phase in phases:
                if phase == 'train':
                    if lr_scheduler is not None:
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)
                        else:
                            lr_scheduler.step()
                        print("current LR: {}".format(
                            optimizer.param_groups[0]['lr']))
                    self.train(True)
                else:
                    self.train(False)

                tic = timer()
                loader = data_loader[phase]
                if isinstance(loader, list):
                    epoch_loss = []
                    epoch_acc = []
                    for ids, train_l in loader:
                        e_loss, e_acc = self.one_step(
                            phase, train_l, optimizer, criterion, use_cuda)
                        epoch_loss.append(e_loss)
                        epoch_acc.append(e_acc)
                    epoch_loss = np.mean(epoch_loss)
                    epoch_acc = np.mean(epoch_acc)
                else:
                    epoch_loss, epoch_acc = self.one_step(
                        phase, loader, optimizer, criterion, use_cuda)
                toc = timer()

                print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.2f}s'.format(
                    phase, epoch_loss, epoch_acc, toc - tic))

                # deep copy the model
                if (phase == 'val') and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch + 1
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            print()

        print('Finish at epoch: {}'.format(epoch + 1))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val loss: {:4f}'.format(best_loss))
        if early_stop:
            self.load_state_dict(best_weights)

        return best_loss, best_acc, best_epoch

    def one_step(self, phase, train_loader, optimizer, criterion, use_cuda):
        running_loss = 0.0
        running_corrects = 0

        for data, target, *_ in train_loader:
            size = data.size(0)
            if self.n_class == 1:
                target = target.float()
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward
            if phase == 'val':
                with torch.no_grad():
                    output = self(data)
                    # print(output.shape)
                    # print(target.shape)
                    # loss = criterion(output, target)
                    if self.n_class == 1:
                        output = output.view(-1)
                        pred = (output.data > 0).float()
                    else:
                        pred = output.data.argmax(dim=1)
                    loss = criterion(output, target)
            else:
                optimizer.zero_grad()
                output = self(data)
                # print(output.shape)
                # print(target.shape)
                # loss = criterion(output, target)
                if self.n_class == 1:
                    output = output.view(-1)
                    pred = (output.data > 0).float()
                else:
                    pred = output.data.argmax(dim=1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                self.normalize_()

            # statistics
            running_loss += loss.item() * size
            running_corrects += torch.sum(pred == target.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        return epoch_loss, epoch_acc

    def hybrid_train(self, teacher_model, train_loader, criterion, optimizer,
                     lr_scheduler=None, init_train_loader=None, epochs=100,
                     val_loader=None, n_sampling_patches=500000,
                     unsup_init=None,
                     use_cuda=False, early_stop=True, regul=1.0):
        print("Initializing CKN layers")
        tic = timer()
        if init_train_loader is not None:
            self.unsup_train_ckn(init_train_loader, n_sampling_patches,
                                 init=unsup_init, use_cuda=use_cuda)
        else:
            self.unsup_train_ckn(train_loader, n_sampling_patches,
                                 init=unsup_init, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

        phases = ['train']
        data_loader = {'train': train_loader}
        if val_loader is not None:
            phases.append('val')
            data_loader['val'] = val_loader

        epoch_loss = None
        best_loss = float('inf')
        best_acc = 0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)
            self.train(False)
            self.hybrid_train_classifier(
                teacher_model, train_loader, criterion,
                use_cuda=use_cuda, regul=regul)

            for phase in phases:
                criterion.weight = None
                criterion.reduction = 'elementwise_mean'
                if phase == 'train':
                    if lr_scheduler is not None:
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)
                        else:
                            lr_scheduler.step()
                        print("current LR: {}".format(
                            optimizer.param_groups[0]['lr']))
                    self.train(True)
                else:
                    self.train(False)

                running_loss = 0.0
                running_corrects = 0

                for data, target, *mask in data_loader[phase]:
                    size = data.size(0)
                    target = target.float()
                    if use_cuda:
                        data = data.cuda()
                        target = target.cuda()
                    if len(mask) > 0:
                        mask = mask[0].view(-1)
                        nu = mask.sum().item()
                        nl = len(mask) - nu
                        weight = torch.ones(len(mask)) / (nl + 1)
                        weight[mask] = regul / (nu + 1)
                        if use_cuda:
                            weight = weight.cuda()
                        criterion.weight = weight
                        criterion.reduction = 'sum'

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == 'val':
                        with torch.no_grad():
                            output = self(data).view(-1)
                            pred = (output > 0).float()
                            loss = criterion(output, target)
                    else:
                        output = self(data).view(-1)
                        pred = (output > 0).float()
                        with torch.no_grad():
                            teacher_pred = teacher_model(data, proba=True).view(-1)
                        target[mask] = (teacher_pred[mask] > 0.5).float()
                        loss = criterion(output, target)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        self.normalize_()

                    # statistics
                    running_loss += loss.item() * size
                    running_corrects += torch.sum(pred == target.data).item()

                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if (phase == 'val') and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            print()

        print('Finish at epoch: {}'.format(epoch + 1))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val loss: {:4f}'.format(best_loss))
        if early_stop:
            self.load_state_dict(best_weights)

        return self

    def hybrid_train_classifier(self, teacher_model, data_loader,
                                criterion=None, use_cuda=False, regul=1.0):
        encoded_train, encoded_target, mask = self.hybrid_predict(
            teacher_model, data_loader, True, use_cuda=use_cuda)
        nu = mask.sum().item()
        nl = len(mask) - nu
        weight = torch.ones(len(encoded_target))
        weight[mask] = regul * nl / (nu + 1)
        if use_cuda:
            weight = weight.cuda()
        criterion.weight = weight
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def hybrid_predict(self, teacher_model, data_loader,
                       only_representation=False,
                       proba=False, use_cuda=False):
        self.train(False)
        if use_cuda:
            self.cuda()
        n_samples = len(data_loader.dataset)
        target_output = torch.Tensor(n_samples)
        mask_output = torch.ByteTensor(n_samples)
        batch_start = 0
        for i, (data, target, mask) in enumerate(data_loader):
            mask = mask.view(-1)
            batch_size = data.shape[0]
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data, proba).data.cpu()
                teacher_target = teacher_model(data, proba=True).data.cpu()
            teacher_target = (teacher_target > 0.5).float()
            teacher_target = teacher_target.view(-1)
            batch_out = torch.cat(
                (batch_out[:batch_size], batch_out[batch_size:]), dim=-1)
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
            output[batch_start:batch_start+batch_size] = batch_out
            target_output[batch_start:batch_start+batch_size] = target
            target_output[batch_start:batch_start+batch_size][mask] = teacher_target[mask]
            mask_output[batch_start:batch_start+batch_size] = mask
            batch_start += batch_size
        output.squeeze_(-1)
        return output, target_output, mask_output

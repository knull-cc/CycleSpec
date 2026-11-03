from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import json
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        # CycleSpec frozen protocol: load a trained base PhaseFormer, freeze it,
        # train only the adapter.
        self.cs_frozen = self.args.model == 'CycleSpec' and \
            getattr(self.args, 'cs_train_mode', 'joint') == 'frozen'
        if self.cs_frozen:
            ckpt = getattr(self.args, 'cs_init_ckpt', '')
            if not ckpt or not os.path.isfile(ckpt):
                raise FileNotFoundError('frozen mode needs --cs_init_ckpt pointing to a base checkpoint, got: {}'.format(ckpt))
            state = torch.load(ckpt, map_location='cpu')
            missing, unexpected = model.load_state_dict(state, strict=False)
            bad_missing = [k for k in missing if not k.startswith('cyclespec.')]
            if bad_missing or unexpected:
                raise RuntimeError('checkpoint does not match the backbone. missing={} unexpected={}'.format(
                    bad_missing, list(unexpected)))
            model.set_backbone_frozen(True)
            print('[CycleSpec] loaded base checkpoint {} and froze the backbone'.format(ckpt))

        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_adapter = sum(p.numel() for p in model.cyclespec.parameters()) if getattr(model, 'use_cyclespec', False) else 0
        self.param_stats = dict(params_total=n_total, params_trainable=n_train, params_adapter=n_adapter,
                                params_backbone=n_total - n_adapter)
        print('[Params] total={} backbone={} adapter={} trainable={}'.format(
            n_total, n_total - n_adapter, n_adapter, n_train))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _probe_batch(self, loader):
        """One fixed input batch for diagnostics. Drawn under a forked RNG so that the
        global RNG stream (data shuffling, dropout) stays identical to a base run."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if not getattr(model, 'use_cyclespec', False):
            return None
        with torch.random.fork_rng(devices=[]):
            batch_x = next(iter(loader))[0]
        return batch_x.float().to(self.device)

    def _delta_ratio(self, probe_x):
        """CycleSpec diagnostic: ||Delta|| / ||X|| on the probe batch."""
        if probe_x is None:
            return None
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        was_training = model.training
        model.eval()
        r = model.cyclespec_delta_ratio(probe_x)
        model.train(was_training)
        return r

    def _select_criterion(self):
        use_huber = float(getattr(self.args, 'use_huber', 0)) > 0
        if use_huber:
            huber_delta = float(getattr(self.args, 'huber_delta', 1.0))
            criterion = nn.HuberLoss(delta=huber_delta)
            print('Using HuberLoss with delta={}'.format(huber_delta))
        else:
            criterion = nn.MSELoss()
            print('Using MSELoss')
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.best_epoch = None
        self.init_vali_loss = None
        self.init_test_loss = None
        probe_x = self._probe_batch(test_loader)
        train_start = time.time()
        if self.cs_frozen:
            # Epoch 0 = the loaded base model (FC_up == 0, so the adapter is the identity).
            # Seeding early stopping with it means the adapter is only kept if it beats
            # the base model on validation.
            self.init_vali_loss = self.vali(vali_data, vali_loader, criterion)
            self.init_test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: 0 (init = base ckpt) | Vali Loss: {0:.7f} Test Loss: {1:.7f}".format(
                self.init_vali_loss, self.init_test_loss))
            early_stopping(self.init_vali_loss, self.model, path)
            self.best_epoch = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            ratio = self._delta_ratio(probe_x)
            if ratio is not None:
                print("\t[CycleSpec] ||Delta||/||X|| on a probe batch: {:.5f}".format(ratio))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.counter == 0:
                self.best_epoch = epoch + 1
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.train_time = time.time() - train_start
        self.best_vali_loss = float(early_stopping.val_loss_min)
        self.epochs_run = epoch + 1
        print("Best epoch: {} | best Vali Loss: {:.7f} | train time: {:.1f}s".format(
            self.best_epoch, self.best_vali_loss, self.train_time))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.ckpt_path = os.path.abspath(best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # one JSON per run, read by scripts/CycleSpec/summarize.py
        a = self.args
        use_cs = a.model == 'CycleSpec'
        summary = dict(
            setting=setting, model=a.model, model_id=a.model_id, data=a.data, data_path=a.data_path,
            seq_len=a.seq_len, pred_len=a.pred_len, seed=getattr(a, 'fix_seed', None),
            variant=('CycleSpec-' + getattr(a, 'cs_train_mode', 'joint')) if use_cs else a.model,
            cs=dict(hidden=a.cs_hidden, windows=a.cs_windows, kernel=a.cs_kernel, branches=a.cs_branches,
                    act=a.cs_act, init_ckpt=a.cs_init_ckpt) if use_cs else None,
            mse=float(mse), mae=float(mae), rmse=float(rmse),
            best_vali_loss=getattr(self, 'best_vali_loss', None), best_epoch=getattr(self, 'best_epoch', None),
            epochs_run=getattr(self, 'epochs_run', None), train_time_s=getattr(self, 'train_time', None),
            init_vali_loss=getattr(self, 'init_vali_loss', None), init_test_loss=getattr(self, 'init_test_loss', None),
            ckpt=getattr(self, 'ckpt_path', None), **getattr(self, 'param_stats', {}),
        )
        with open(os.path.join(folder_path, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        return

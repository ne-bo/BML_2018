import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from tqdm import tqdm

from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizers_and_schedulers, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizers_and_schedulers, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(self.batch_size))
        self.optimizers_and_schedulers = optimizers_and_schedulers

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        batch_size = self.data_loader.batch_size
        for batch_idx, (data, target) in tqdm(enumerate(self.data_loader)):
            x, target = data.to(self.device), target.to(self.device)

            x = Variable(x, requires_grad=True).to(self.device)

            # // AAE phase
            # z ∼ p(z)
            # zc ← CG(z)
            # xrec ← dec(enc(x))

            z = torch.randn(batch_size, self.model.noise_size, 1, 1, requires_grad=True, device=self.device)
            d_c_enc_x, d_c_z_c, x_rec = self.model(x, z, phase='AAE')

            l_c, l_combined, l_rec = self.calculate_losses_for_AAE_phase(d_c_enc_x, d_c_z_c, x, x_rec)

            self.update_AAE_phase(l_c, l_combined, l_rec)

            # // Prior improvement phase
            # z ∼ p(z)
            # zc ← CG(z)
            # xnoise ← dec(zc)
            # xrec ← dec(enc(xj ))
            z = torch.randn(batch_size, self.model.noise_size, 1, 1, requires_grad=True, device=self.device)
            d_i_x, d_i_dec_z_c, d_i_x_rec = self.model(x, z, phase='PriorImprovement')

            l_i, minus_l_i = self.calculate_losses_for_prior_improvement_phase(d_i_dec_z_c, d_i_x, d_i_x_rec)

            self.update_prior_improvement_phase(l_i, minus_l_i)

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', l_i.item())
            total_loss += l_i.item()
            # total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    l_i.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.optimizers_and_schedulers is not None:
            self.optimizers_and_schedulers.encoder_scheduler.step()
            self.optimizers_and_schedulers.decoder_scheduler.step()
            self.optimizers_and_schedulers.code_generator_scheduler.step()
            self.optimizers_and_schedulers.d_i_scheduler.step()
            self.optimizers_and_schedulers.d_c_scheduler.step()

        return log

    def calculate_losses_for_prior_improvement_phase(self, d_i_dec_z_c, d_i_x, d_i_x_rec):
        # LIGAN ← log(DI (xj )) + log(1 − DI (xnoise)) + log(1 − DI (xrec))
        l_i = self.loss.l_i(d_i_x, d_i_dec_z_c, d_i_x_rec)
        minus_l_i = -l_i
        return l_i, minus_l_i

    def calculate_losses_for_AAE_phase(self, d_c_enc_x, d_c_z_c, x, x_rec):
        # LC GAN ← log(DC (zc)) + log(1 − DC (enc(x)))
        l_c = self.loss.l_c(d_c_enc_x, d_c_z_c)
        # Lrec ← 1NkF(x) − F(xrec)k2
        l_rec = self.loss.l_rec(x, x_rec)
        l_combined = -l_c + l_rec
        return l_c, l_combined, l_rec

    def update_prior_improvement_phase(self, l_i, minus_l_i):
        # // Update network parameters for prior improvement phase
        # θDI ← θDI − ∇θDI(LIGAN )
        #####################################
        # I think that code_generator should be updated here using l_i loss
        #####################################
        self.optimizers_and_schedulers.d_i_optimizer.zero_grad()
        self.optimizers_and_schedulers.code_generator_optimizer.zero_grad()
        l_i.backward(retain_graph=True)
        self.optimizers_and_schedulers.d_i_optimizer.step()
        self.optimizers_and_schedulers.code_generator_optimizer.step()
        # θdec ← θdec − ∇θdec (−LIGAN )
        self.optimizers_and_schedulers.decoder_optimizer.zero_grad()
        minus_l_i.backward()
        self.optimizers_and_schedulers.decoder_optimizer.step()

    def update_AAE_phase(self, l_c, l_combined, l_rec):
        # // Update network parameters for AAE phase
        # θDC ← θDC − ∇θDC(LCGAN )
        self.optimizers_and_schedulers.d_c_optimizer.zero_grad()
        l_c.backward(retain_graph=True)
        self.optimizers_and_schedulers.d_c_optimizer.step()
        # θenc ← θenc − ∇θenc(−LCGAN + Lrec)
        self.optimizers_and_schedulers.encoder_optimizer.zero_grad()
        l_combined.backward(retain_graph=True)
        self.optimizers_and_schedulers.encoder_optimizer.step()
        # θdec ← θdec − ∇θdec (λ ∗ Lrec)
        self.optimizers_and_schedulers.decoder_optimizer.zero_grad()
        l_rec.backward()
        self.optimizers_and_schedulers.decoder_optimizer.step()

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

from copy import deepcopy
import functools

import numpy as np
import torch as t
import torch.nn as nn
import jukebox.utils.dist_adapter as dist
from tqdm import tqdm

from jukebox.transformer.ops import LayerNorm
from jukebox.prior.autoregressive import ConditionalAutoregressive2D
from jukebox.prior.conditioners import Conditioner, LabelConditioner
from jukebox.data.labels import EmptyLabeller, Labeller

from jukebox.utils.torch_utils import assert_shape
from jukebox.utils.dist_utils import print_once
from jukebox.vqvae.vqvae import calculate_strides
from jukebox.prior.autoregressive import roll
from jukebox.utils.checkpoint import checkpoint


"""
Model the prior on vq codes conditioned on timing, artist, genre, lyrics and codes from levels above. 
To condition on the timing, genre and artist, we use the LabelConditioner class
To condition on the codes from the level above, we use the Conditioner class
To condition on lyrics, we allow two types of priors:
- Separate Encoder Decoder: This is the usual encoder-decoder style transformer. The encoder transformer autoregressively 
models the lyrics, and we use its last layer to produce keys/values that are attened to by the decoder transformer
- Single Encoder Decoder: This is a simplification where we combine them into a single model. We merge the text vocab 
and VQ vocab into a single large vocab, and the lyric tokens and VQ tokens into a single longer sequence of tokens which 
we autoregressively model together.
"""
class SimplePrior(nn.Module):
    def __init__(self, z_shapes, l_bins, encoder, decoder, level,
                 downs_t, strides_t, labels, prior_kwargs, x_cond_kwargs, y_cond_kwargs,
                 prime_kwargs, copy_input, labels_v3=False,
                 merged_decoder=False, single_enc_dec=False, debug=False):
        super().__init__()
        self.debug= debug
        self.use_tokens = prime_kwargs.pop('use_tokens')
        self.n_tokens = prime_kwargs.pop('n_tokens')
        self.prime_loss_fraction = prime_kwargs.pop('prime_loss_fraction')

        self.copy_input = copy_input
        if self.copy_input:
            prime_kwargs['bins'] = l_bins

        self.z_shapes = z_shapes
        self.levels = len(self.z_shapes)

        self.z_shape = self.z_shapes[level]

        self.level = level
        assert level < self.levels, f"Total levels {self.levels}, got level {level}"

        self.l_bins = l_bins

        # Passing functions instead of the vqvae module to avoid getting params
        self.encoder = encoder
        self.decoder = decoder

        # X conditioning
        self.x_cond = (level != (self.levels - 1))
        self.cond_level = level + 1

        # Y conditioning
        self.y_cond = labels

        self.single_enc_dec = single_enc_dec
        # X conditioning
        if self.x_cond:
            self.conditioner_blocks = nn.ModuleList()
            conditioner_block = lambda _level: Conditioner(input_shape=z_shapes[_level],
                                                          bins=l_bins,
                                                          down_t=downs_t[_level],
                                                          stride_t=strides_t[_level],
                                                          **x_cond_kwargs)
            if dist.get_rank() == 0: print(f"Conditioning on 1 above level(s)")
            self.conditioner_blocks.append(conditioner_block(self.cond_level))

        # Y conditioning
        if self.y_cond:
            self.n_time = self.z_shape[0] # Assuming STFT=TF order and raw=T1 order, so T is first dim
            self.y_emb = LabelConditioner(n_time=self.n_time,include_time_signal=not self.x_cond,**y_cond_kwargs)

        # Lyric conditioning
        if single_enc_dec:
            # Single encoder-decoder transformer
            self.prior_shapes = [(self.n_tokens,), prior_kwargs.pop('input_shape')]
            self.prior_bins = [prime_kwargs['bins'], prior_kwargs.pop('bins')]
            self.prior_dims = [np.prod(shape) for shape in self.prior_shapes]
            self.prior_bins_shift = np.cumsum([0, *self.prior_bins])[:-1]
            self.prior_width = prior_kwargs['width']
            print_once(f'Creating cond. autoregress with prior bins {self.prior_bins}, ')
            print_once(f'dims {self.prior_dims}, ')
            print_once(f'shift {self.prior_bins_shift}')
            print_once(f'input shape {sum(self.prior_dims)}')
            print_once(f'input bins {sum(self.prior_bins)}')
            print_once(f'Self copy is {self.copy_input}')

            self.prime_loss_dims, self.gen_loss_dims = self.prior_dims[0], self.prior_dims[1]
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(input_shape=(sum(self.prior_dims),),
                                                     bins=sum(self.prior_bins),
                                                     x_cond=(self.x_cond or self.y_cond), y_cond=True,
                                                     prime_len=self.prime_loss_dims,
                                                     **prior_kwargs)

        else:
            # Separate encoder-decoder transformer
            if self.n_tokens != 0 and self.use_tokens:
                from jukebox.transformer.ops import Conv1D
                prime_input_shape = (self.n_tokens,)
                self.prime_loss_dims = np.prod(prime_input_shape)
                self.prime_acts_width, self.prime_state_width = prime_kwargs['width'], prior_kwargs['width']
                self.prime_prior = ConditionalAutoregressive2D(input_shape=prime_input_shape, x_cond=False, y_cond=False,
                                                               only_encode=True,
                                                               **prime_kwargs)
                self.prime_state_proj = Conv1D(self.prime_acts_width, self.prime_state_width, init_scale=prime_kwargs['init_scale'])
                self.prime_state_ln = LayerNorm(self.prime_state_width)
                self.prime_bins = prime_kwargs['bins']
                self.prime_x_out = nn.Linear(self.prime_state_width, self.prime_bins, bias=False)
                nn.init.normal_(self.prime_x_out.weight, std=0.02 * prior_kwargs['init_scale'])
            else:
                self.prime_loss_dims = 0
            self.gen_loss_dims = np.prod(self.z_shape)
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(x_cond=(self.x_cond or self.y_cond), y_cond=self.y_cond,
                                                     encoder_dims = self.prime_loss_dims, merged_decoder=merged_decoder,
                                                     **prior_kwargs)

        self.n_ctx = self.gen_loss_dims
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.cond_downsample = self.downsamples[level+1] if level != self.levels - 1 else None
        self.raw_to_tokens = np.prod(self.downsamples[:level+1])
        self.sample_length = self.n_ctx*self.raw_to_tokens
        if labels:
            self.labels_v3 = labels_v3
            self.labeller = Labeller(self.y_emb.max_bow_genre_size, self.n_tokens, self.sample_length, v3=self.labels_v3)
        else:
            self.labeller = EmptyLabeller()

        print(f"Level:{level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample length:{self.sample_length}")


    def get_y(self, labels, start, get_indices=False):
        if isinstance(self.labeller, EmptyLabeller):
            return None
        y = labels['y'].clone()

        # Set sample_length to match this level
        y[:, 2] = int(self.sample_length)

        # Set offset
        y[:, 1:2] = y[:, 1:2] + int(start * self.raw_to_tokens)

        # Set lyric tokens
        indices = self.labeller.set_y_lyric_tokens(y, labels)
        if get_indices:
            return y, indices
        else:
            return y

    def get_z_conds(self, zs, start, end):
        if self.level != self.levels - 1:
            assert start % self.cond_downsample == end % self.cond_downsample == 0
            z_cond = zs[self.level + 1][:,start//self.cond_downsample:end//self.cond_downsample]
            assert z_cond.shape[1] == self.n_ctx//self.cond_downsample
            z_conds = [z_cond]
        else:
            z_conds = None
        return z_conds

    def prior_preprocess(self, xs, conds):
        N = xs[0].shape[0]
        for i in range(len(xs)):
            x, shape, dims = xs[i], self.prior_shapes[i], self.prior_dims[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            assert isinstance(x, t.cuda.LongTensor), x
            assert (0 <= x).all() and (x < bins).all()
            #assert_shape(x, (N, *shape))
            xs[i] = (xs[i] + bins_shift).view(N, -1)

        for i in range(len(conds)):
            cond, shape, dims = conds[i], self.prior_shapes[i], self.prior_dims[i]
            if cond is not None:
                assert_shape(cond, (N, dims, self.prior_width))
            else:
                conds[i] = t.zeros((N, dims, self.prior_width), dtype=t.float, device='cuda')

        return t.cat(xs, dim=1), t.cat(conds, dim=1)

    def prior_postprocess(self, z):
        N = z.shape[0]
        dims = (self.prior_dims[0], z.shape[1] - self.prior_dims[0])
        # xs = list(t.split(z, self.prior_dims, dim=1))
        xs = list(t.split(z, dims, dim=1))

        for i in range(len(xs)):
            # x, shape, dims, bins, bins_shift = xs[i], self.prior_shapes[i], self.prior_dims[i], self.prior_bins[i], self.prior_bins_shift[i]
            # assert_shape(x, (N, dims))
            shape = self.prior_shapes[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            # xs[i] = (xs[i] - bins_shift).view(N, *shape) #view(N, -1, *shape[1:])
            xs[i] = (xs[i] - bins_shift).view(N, -1, *shape[1:])
            xs[i] = t.clamp(xs[i], min=0)  # If not masking loss, model may have generated lyric/midi tokens which are now shifted <0 by bin_shift
            assert (xs[i] < bins).all(), f'rank: {dist.get_rank()}, bins: {bins}, dims {dims}, shape {shape}, prior_shape {self.prior_shapes}, bins_shift {bins_shift}, xs[i]: {xs[i]}'

        return xs[-1]

    def x_emb(self, z_conds):
        z_conds = z_conds[:self.cond_level - self.level]
        assert len(z_conds) == len(self.conditioner_blocks) == self.cond_level - self.level, f"Expected {len(z_conds)} == {len(self.conditioner_blocks)} == {self.cond_level} - {self.level}"
        x_cond = None
        for z_cond, conditioner_block in reversed(list(zip(z_conds, self.conditioner_blocks))):
            x_cond = conditioner_block(z_cond, x_cond)
        return x_cond

    def encode(self, x, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        # Get latents
        with t.no_grad():
            zs = self.encoder(x, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return zs

    def decode(self, zs, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels

        assert len(zs) == end_level - start_level
        with t.no_grad():
            x_out = self.decoder(zs, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return x_out

    def get_cond(self, z_conds, y):
        if y is not None:
            assert y.shape[1] == 4 + self.y_emb.max_bow_genre_size + self.n_tokens, f"Expected {4} + {self.y_emb.max_bow_genre_size} + {self.n_tokens}, got {y.shape[1]}"
            n_labels = y.shape[1] - self.n_tokens
            y, prime = y[:,:n_labels], y[:,n_labels:]
        else:
            y, prime = None, None
        y_cond, y_pos = self.y_emb(y) if self.y_cond else (None, None)
        x_cond = self.x_emb(z_conds) if self.x_cond else y_pos
        return x_cond, y_cond, prime

    def sample(self, n_samples, z=None, z_conds=None, y=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               chunk_size=None, sample_tokens=None):
        N = n_samples
        if z is not None: assert z.shape[0] == N, f"Expected shape ({N},**), got shape {z.shape}"
        if y is not None: assert y.shape[0] == N, f"Expected shape ({N},**), got shape {y.shape}"
        if z_conds is not None:
            for z_cond in z_conds:
                assert z_cond.shape[0] == N,  f"Expected shape ({N},**), got shape {z_cond.shape}"

        no_past_context = (z is None or z.shape[1] == 0)
        if dist.get_rank() == 0:
            name = {True: 'Ancestral', False: 'Primed'}[no_past_context]
            print(f"{name} sampling {n_samples} samples with temp={temp}, top_k={top_k}, top_p={top_p}")

        with t.no_grad():
            # Currently x_cond only uses immediately above layer
            x_cond, y_cond, prime = self.get_cond(z_conds, y)
            if self.single_enc_dec:
                # assert chunk_size % self.prime_loss_dims == 0. TODO: Check if needed
                if no_past_context:
                    z, x_cond = self.prior_preprocess([prime], [None, x_cond])
                else:
                    z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
                if sample_tokens is not None:
                    sample_tokens += self.n_tokens
                z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, fp16=fp16, temp=temp,
                                             top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
                z = self.prior_postprocess(z)
            else:
                encoder_kv = self.get_encoder_kv(prime, fp16=fp16, sample=True)
                if no_past_context:
                    z = self.prior.sample(n_samples, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k,
                                          top_p=top_p, sample_tokens=sample_tokens)
                else:
                    z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp,
                                             top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
            if sample_tokens is None:
                assert_shape(z, (N, *self.z_shape))
        return z

    def get_encoder_kv(self, prime, fp16=False, sample=False):
        if self.n_tokens != 0 and self.use_tokens:
            if sample:
                self.prime_prior.cuda()
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, None, None, None, fp16=fp16)
            assert_shape(prime_acts, (N, self.prime_loss_dims, self.prime_acts_width))
            assert prime_acts.dtype == t.float, f'Expected t.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert encoder_kv.dtype == t.float, f'Expected t.float, got {encoder_kv.dtype}'
            if sample:
                self.prime_prior.cpu()
                if fp16:
                    encoder_kv = encoder_kv.half()
        else:
            encoder_kv = None
        return encoder_kv

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, self.prime_bins), prime_t.view(-1)) / np.log(2.)
        else:
            prime_loss = t.tensor(0.0, device='cuda')
        return prime_loss

    def z_forward(self, z, z_conds=[], y=None, fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:,:self.n_tokens]
        if self.single_enc_dec: # True for the top level prior
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
            (prime_loss, gen_loss), preds = self.prior(z, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        else:
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            prime_loss = self.get_prime_loss(encoder_kv, prime)
            gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=get_preds)
        loss = (self.prime_loss_fraction*prime_loss*self.prime_loss_dims/self.total_loss_dims) + \
                   (gen_loss*self.gen_loss_dims/self.total_loss_dims)
        metrics=dict(bpd=gen_loss.clone().detach(), prime_loss=prime_loss.clone().detach(),
                     gen_loss=gen_loss.clone().detach())
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics

    def forward(self, x, y=None, fp16=False, decode=False, get_preds=False):
        bs = x.shape[0]
        z, *z_conds = self.encode(x, bs_chunks=bs) # [bs, 8192], []
        loss, metrics = self.z_forward(z=z, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        if decode:
            x_out = self.decode([z, *z_conds])
        else:
            x_out = None
        return x_out, loss, metrics
    
    def finetune_forward(self, z, pred_mask, sep_mask, pad_mask, y, fp16=False, decode=False, get_preds=False):
        z_conds = []
        loss, metrics = self.finetune_z_forward(z, pred_mask, sep_mask, pad_mask, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        if decode:
            x_out = self.decode([z, *z_conds])
        else:
            x_out = None
        return x_out, loss, metrics
    
    def finetune_z_forward(self, z, pred_mask, sep_mask, pad_mask, y=None, z_conds=[], fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:,:self.n_tokens]
        if self.single_enc_dec: # True for the top level prior
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])

            # Account for the left-concatted cond
            sep_mask = t.cat((t.zeros_like(prime), sep_mask), dim=1)
            pad_mask = t.cat((t.zeros_like(prime), pad_mask), dim=1)

            (prime_loss, gen_loss), preds = self.prior.finetune_forward(z, pred_mask, sep_mask, pad_mask, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        else:
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            prime_loss = self.get_prime_loss(encoder_kv, prime)
            gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=get_preds)
        loss = (self.prime_loss_fraction*prime_loss*self.prime_loss_dims/self.total_loss_dims) + \
                   (gen_loss*self.gen_loss_dims/self.total_loss_dims)
        metrics=dict(bpd=gen_loss.clone().detach(), prime_loss=prime_loss.clone().detach(),
                     gen_loss=gen_loss.clone().detach())
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics
        
    def finetune_sample_z(self, z, pred_mask, sep_mask, pad_mask, y=None, z_conds=[], fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:,:self.n_tokens]
        if self.single_enc_dec: # True for the top level prior
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])

            # Account for the left-concatted cond
            sep_mask = t.cat((t.zeros_like(prime), sep_mask), dim=1)
            pad_mask = t.cat((t.zeros_like(prime), pad_mask), dim=1)
            pred_mask = t.cat((t.zeros_like(prime), pred_mask), dim=1)

            z_pred, z_true = self.prior.finetune_sample(z, pred_mask, sep_mask, pad_mask, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
            
            # Account for bins_shift
            z_pred = z_pred - self.prior_bins_shift[1]
            z_true = z_true - self.prior_bins_shift[1]

            return z_pred, z_true
    
    def initialize_controlnet(self):
        # Copy the top transformer prior (the weights will be copied later)
        self.prior_copy = deepcopy(self.prior)

        # Initialize zero convs
        zero_convs = []
        for _ in range(len(self.prior.transformer._attn_mods) + 1):
            if self.debug:
                conv = t.nn.utils.skip_init(t.nn.Conv2d, 1, 1, 1) # channel_in, channel_out, kernel_size
            else:
                conv = t.nn.utils.skip_init(t.nn.Conv2d, 1, 1, 1, dtype=t.half).cuda() # channel_in, channel_out, kernel_size
            t.nn.init.constant_(conv.weight, 0)
            t.nn.init.constant_(conv.bias, 0)
            zero_convs.append(conv)
        self.zero_convs = t.nn.ModuleList(zero_convs)
        return
    
    def controlnet_copy_params(self):
        self.prior_copy.load_state_dict(self.prior.state_dict())

    def controlnet_forward(self, z_src, z_tar, pred_mask, pad_mask, y, fp16=False, decode=False, get_preds=False):
        z_conds = []
        loss, metrics = self.controlnet_z_forward(z_src, z_tar, pred_mask, pad_mask, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        
        x_out = None
        return x_out, loss, metrics

    def controlnet_z_forward(self, z_src, z_tar, pred_mask, pad_mask, y=None, z_conds=[], fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        z_src, x_cond = self.prior_preprocess([prime, z_src], [None, x_cond])
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        z_tar, x_cond = self.prior_preprocess([prime, z_tar], [None, x_cond])

        # Account for the left-concatted cond
        pad_mask = t.cat((t.zeros_like(prime), pad_mask), dim=1)

        (prime_loss, gen_loss), preds = self.controlnet_prior_forward(z_src, z_tar, pred_mask, pad_mask, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)

        loss = (self.prime_loss_fraction*prime_loss*self.prime_loss_dims/self.total_loss_dims) + \
                   (gen_loss*self.gen_loss_dims/self.total_loss_dims)
        metrics=dict(bpd=gen_loss.clone().detach(), prime_loss=prime_loss.clone().detach(),
                     gen_loss=gen_loss.clone().detach())
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics
        
    def controlnet_prior_forward(self, z_src, z_tar, pred_mask, pad_mask, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False,
                encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        # Pprocess.
        with t.no_grad():
            z_src = self.prior_copy.preprocess(z_src)
            z_tar = self.prior.preprocess(z_tar)

        N, D = z_src.shape
        
        for x in [z_src, z_tar]:
            assert isinstance(x, t.cuda.LongTensor)
            assert (0 <= x).all() and (x < self.prior.bins).all()

        if self.prior.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.prior.width)
        else:
            assert y_cond is None

        if self.prior.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.prior.width) or x_cond.shape == (N, 1, self.prior.width), f"{x_cond.shape} != {(N, D, self.prior.width)} nor {(N, 1, self.prior.width)}. Did you pass the correct --sample_length?"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.prior.width), device=x.device, dtype=t.float)

        z_tar_t = z_tar # Target
        z_src = self.prior_copy.x_emb(z_src) # X emb
        z_tar = self.prior.x_emb(z_tar)
        z_src = roll(z_src, 1) # Shift by 1
        z_tar = roll(z_tar, 1)

        # Also shift the masks
        pad_mask = roll(pad_mask, 1)

        # Apply the pad masks
        z_src[pad_mask == 1] = self.prior_copy.pad_token
        z_tar[pad_mask == 1] = self.prior.pad_token
        
        # Fill in start token. The first token is always pad and can be safely replaced.
        if self.prior.y_cond:
            z_src[:,0] = y_cond.view(N, self.prior_copy.width)
            z_tar[:,0] = y_cond.view(N, self.prior.width)
        else:
            z_src[:,0] = self.prior_copy.start_token
            z_tar[:,0] = self.prior.start_token

        z_src = self.prior_copy.x_emb_dropout(z_src) + self.prior_copy.pos_emb_dropout(self.prior_copy.pos_emb()) + x_cond # Pos emb and dropout
        z_tar = self.prior.x_emb_dropout(z_tar) + self.prior.pos_emb_dropout(self.prior.pos_emb()) + x_cond # Pos emb and dropout

        x = self.controlnet_transformer_forward(z_src, z_tar, encoder_kv=encoder_kv, fp16=fp16) # Transformer

        if self.debug:
            x_cond = x_cond.cpu()
            pred_mask = pred_mask.cpu()
            z_tar_t = z_tar_t.cpu()

        if self.prior.add_cond_after_transformer: # Piped doesnt add x_cond
            x = x + x_cond

        acts = x
        if self.prior.only_encode:
            return x
        x = self.prior.x_out(x) # Predictions

        assert self.prior.prime_len is not None
        x_gen = x[:, self.prior.prime_len:][pred_mask == 1].reshape(-1, self.prior.bins)

        gen_loss = t.nn.functional.cross_entropy(x_gen, z_tar_t[:, self.prior.prime_len:][pred_mask == 1].reshape(-1)) / np.log(2.)
        prime_loss = t.zeros_like(gen_loss)

        loss = (prime_loss, gen_loss) # Note order! Prime is first

        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None
        
    def controlnet_transformer_forward(self, z_src, z_tar, encoder_kv=None, sample=False, fp16=False, fp16_out=False):
        if self.debug:
            self.cpu()
            z_src = z_src.cpu()
            z_tar = z_tar.cpu()
        elif fp16:
            z_src = z_src.half()
            z_tar = z_tar.half()

        # Blocks
        for i,l in enumerate(self.prior.transformer._attn_mods):
            # Zero conv for the inputs
            if i == 0:
                z_src = z_src + t.squeeze(self.zero_convs[0](t.unsqueeze(z_tar, 1)), 1)

            # Frozen copy
            f = functools.partial(l, encoder_kv=None, sample=sample)
            z_tar = checkpoint(f, (z_tar,), l.parameters(), True)

            # Trainable (condition) copy
            l = self.prior_copy.transformer._attn_mods[i]
            f = functools.partial(l, encoder_kv=None, sample=sample)
            z_src = checkpoint(f, (z_src,), l.parameters(), True)

            # Zero conv
            z_tar = z_tar + t.squeeze(self.zero_convs[i + 1](t.unsqueeze(z_src, 1)), 1)
        if not fp16_out:
            z_tar = z_tar.float()
        return z_tar
    
    def controlnet_sample_z(self, z_src, z_tar, pred_mask, pad_mask, y=None, z_conds=[], fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        z_src, x_cond = self.prior_preprocess([prime, z_src], [None, x_cond])
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        z_tar, x_cond = self.prior_preprocess([prime, z_tar], [None, x_cond])

        # Account for the left-concatted cond
        pad_mask = t.cat((t.zeros_like(prime), pad_mask), dim=1)
        pred_mask = t.cat((t.zeros_like(prime), pred_mask), dim=1)
        
        z_pred, z_true = self.controlnet_prior_sample(z_src, z_tar, pred_mask, pad_mask, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        
        # Account for bins_shift
        z_pred = z_pred - self.prior_bins_shift[1]
        z_true = z_true - self.prior_bins_shift[1]

        return z_pred, z_true
    
    def controlnet_prior_sample(self, z_src, z_tar, pred_mask, pad_mask, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False,
                encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        # Pprocess.
        with t.no_grad():
            z_src = self.prior_copy.preprocess(z_src)
            z_tar = self.prior.preprocess(z_tar)

            N, D = z_src.shape
            
            for x in [z_src, z_tar]:
                assert isinstance(x, t.cuda.LongTensor)
                assert (0 <= x).all() and (x < self.prior.bins).all()

            if self.prior.y_cond:
                assert y_cond is not None
                assert y_cond.shape == (N, 1, self.prior.width)
            else:
                assert y_cond is None

            if self.prior.x_cond:
                assert x_cond is not None
                assert x_cond.shape == (N, D, self.prior.width) or x_cond.shape == (N, 1, self.prior.width), f"{x_cond.shape} != {(N, D, self.prior.width)} nor {(N, 1, self.prior.width)}. Did you pass the correct --sample_length?"
            else:
                assert x_cond is None
                x_cond = t.zeros((N, 1, self.prior.width), device=x.device, dtype=t.float)

            z_tar_t = z_tar # Target
            z_src = roll(z_src, 1) # Shift by 1
            z_tar = roll(z_tar, 1)

            # Also shift the masks
            pad_mask = roll(pad_mask, 1)

            # At test time, mask off all the target z sequence
            pad_mask[pred_mask == 1] = 1

            # Locate the span to predict
            for i in range(pred_mask.shape[1]):
                if pred_mask[0, i] == 1:
                    start_idx = i
                    break

            # Autoregressive sampling
            for i in tqdm(range(t.sum(pred_mask).int())):
                pred_idx =  start_idx + i

                # Remove the pad mask at the current token
                pad_mask[:, pred_idx] = 0

                # Make embs 
                emb_src = self.prior_copy.x_emb(z_src)
                emb_tar = self.prior.x_emb(z_tar)

                # Apply the pad masks
                emb_src[pad_mask == 1] = self.prior_copy.pad_token
                emb_tar[pad_mask == 1] = self.prior.pad_token
            
                # Fill in start token. The first token is always pad and can be safely replaced.
                if self.prior.y_cond:
                    emb_src[:,0] = y_cond.view(N, self.prior_copy.width)
                    emb_tar[:,0] = y_cond.view(N, self.prior.width)
                else:
                    emb_src[:,0] = self.prior_copy.start_token
                    emb_tar[:,0] = self.prior.start_token

                emb_src = self.prior_copy.x_emb_dropout(emb_src) + self.prior_copy.pos_emb_dropout(self.prior_copy.pos_emb()) + x_cond # Pos emb and dropout
                emb_tar = self.prior.x_emb_dropout(emb_tar) + self.prior.pos_emb_dropout(self.prior.pos_emb()) + x_cond # Pos emb and dropout

                x = self.controlnet_transformer_forward(emb_src, emb_tar, encoder_kv=encoder_kv, fp16=fp16) # Transformer

                if self.debug:
                    x_cond = x_cond.cpu()
                    pred_mask = pred_mask.cpu()
                    z_tar_t = z_tar_t.cpu()

                if self.prior.add_cond_after_transformer: # Piped doesnt add x_cond
                    x = x + x_cond

                pred = self.prior.x_out(x) # Predictions

                pred = pred[:, pred_idx, :] 
                # Argmax samping
                # pred = t.argmax(pred[0])

                # Sample by softmax
                pred = t.nn.Softmax(dim=1)(pred)[0]
                pred = t.multinomial(pred, 1)[0]
                
                if i < 40:
                    print(pred)
                    print(z_tar_t[0, pred_idx])
                
                # Update the input sequence
                if i == z_tar.shape[1] - 1:
                    z_tar[0, 0] = pred
                else:
                    z_tar[0, pred_idx+1] = pred
            
            # Return the predicted z sequence
            z_tar = roll(z_tar, -1)
            z_pred = z_tar[:, start_idx: start_idx+t.sum(pred_mask).int()]
            z_true = z_tar_t[:, start_idx: start_idx+t.sum(pred_mask).int()]

            return z_pred, z_true

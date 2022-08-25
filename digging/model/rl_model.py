import torch
import torch.nn as nn
from typing import Any, Optional, Sequence, Union, Dict

from ding.model.common import ReparameterizationHead, DiscreteHead, MultiHead, DuelingHead, RegressionHead
from .rnn_model import LSTMSeqModel, GRUSeqModel


class RNNDiscretePGModel(nn.Module):
    """
    Policy Gradient model with RNN encoder for RL searching engines on discrete searching space.
    """

    def __init__(
            self,
            obs_shape: int,
            action_shape: int,
            encoder_hidden_size: int,
            sequence_len: int,
            num_layers: int = 1,
            model_type: str = 'lstm',
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self.input_layer = nn.Embedding(self._obs_shape, encoder_hidden_size * num_layers)
        assert model_type in ['lstm', 'gru'], model_type
        if model_type == 'lstm':
            self.encoder = LSTMSeqModel(encoder_hidden_size, head_hidden_size, sequence_len, num_layers)
        elif model_type == 'gru':
            self.encoder = GRUSeqModel(encoder_hidden_size, head_hidden_size, sequence_len, num_layers)

        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def forward(self, x: torch.Tensor) -> Dict:
        x = self.input_layer(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x, h = self.encoder(x, None)
        x = self.head(x)
        return x


class RNNContinuousPGModel(nn.Module):
    """
    Policy Gradient model with RNN encoder for RL searching engines on continuous searching space.
    """

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        encoder_hidden_size: int,
        sequence_len: int,
        action_space: str = 'regression',
        num_layers: int = 1,
        model_type: str = 'lstm',
        head_hidden_size: Optional[int] = None,
        head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        bound_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._action_space = action_space
        self.input_layer = nn.Linear(self._obs_shape, encoder_hidden_size * num_layers)
        assert model_type in ['lstm', 'gru'], model_type
        if model_type == 'lstm':
            self.encoder = LSTMSeqModel(encoder_hidden_size, head_hidden_size, sequence_len, num_layers)
        elif model_type == 'gru':
            self.encoder = GRUSeqModel(encoder_hidden_size, head_hidden_size, sequence_len, num_layers)

        assert self._action_space in ['regression', 'reparameterization']
        if self._action_space == 'regression':
            self.head = RegressionHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                final_tanh=True,
                activation=activation,
                norm_type=norm_type
            )
        elif self._action_space == 'reparameterization':
            self.head = ReparameterizationHead(
                head_hidden_size,
                action_shape,
                head_layer_num,
                sigma_type='conditioned',
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type
            )

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        x = self.input_layer(inputs)
        x = x.unsqueeze(1)
        x, h = self.encoder(x, None)
        if self._action_space == 'regression':
            x = self.head(x)
            return {'action': x['pred']}
        elif self._action_space == 'reparameterization':
            x = self.head(x)
            x = {k: v.squeeze() for k, v in x.items()}
            return {'logit': x}


class RNNVACModel(nn.Module):
    """
    PPO model with RNN encoder for RL searching engines.
    """

    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        sequence_len: int,
        num_layers: int = 1,
        encoder_hidden_size: int = 64,
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        model_type: str = 'lstm',
        share_encoder: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        bound_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._obs_shape, self._action_shape = obs_shape, action_shape
        self._action_space = action_space
        assert self._action_space in ['discrete', 'continuous'], self.action_space
        self._share_encoder = share_encoder

        def make_encoder(out_size):
            if self._action_space == 'discrete':
                input_layer = nn.Embedding(self._obs_shape, encoder_hidden_size * num_layers)
            else:
                input_layer = nn.Linear(self._obs_shape, encoder_hidden_size * num_layers)
            assert model_type in ['lstm', 'gru'], model_type
            if model_type == 'lstm':
                encoder = LSTMSeqModel(encoder_hidden_size, out_size, sequence_len, num_layers)
            elif model_type == 'gru':
                encoder = GRUSeqModel(encoder_hidden_size, out_size, sequence_len, num_layers)
            return nn.Sequential(input_layer, encoder)

        if self._share_encoder:
            self.encoder = make_encoder(actor_head_hidden_size)
        else:
            self.actor_encoder = make_encoder(actor_head_hidden_size)
            self.critic_encoder = make_encoder(critic_head_hidden_size)
        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )
        if self._action_space == 'continuous':
            self.multi_head = False
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type
            )
        elif self._action_space == 'discrete':
            actor_head_cls = DiscreteHead
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head
            if multi_head:
                self.actor_head = MultiHead(
                    actor_head_cls,
                    actor_head_hidden_size,
                    action_shape,
                    layer_num=actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                self.actor_head = actor_head_cls(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )

        # must use list, not nn.ModuleList
        if self._share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        if self._share_encoder:
            x, h = self.encoder(x)
        else:
            x, h = self.actor_encoder(x)

        if self._action_space == 'discrete':
            x = self.actor_head(x)
            return {'logit': x['logit'].squeeze()}
        elif self._action_space == 'continuous':
            x = self.actor_head(x)  # mu, sigma
            return {'logit': {'mu': x['mu'].squeeze(), 'sigma': x['sigma'].squeeze()}}

    def compute_critic(self, x: torch.Tensor) -> Dict:
        if self._share_encoder:
            x, h = self.encoder(x)
        else:
            x, h = self.critic_encoder(x)
        if isinstance(h, tuple):
            h = torch.cat([h[0], h[1]], dim=-1)
        x = h.reshape((x.shape[0], -1))
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: torch.Tensor) -> Dict:
        if self._share_encoder:
            actor_embedding, actor_h = critic_embedding, critic_h = self.encoder(x)
        else:
            actor_embedding, actor_h = self.actor_encoder(x)
            critic_embedding, critic_h = self.critic_encoder(x)

        if isinstance(critic_h, tuple):
            critic_h = torch.cat([critic_h[0], critic_h[1]], dim=-1)
        critic_embedding = critic_h.reshape(critic_embedding.shape[0], -1)
        value = self.critic_head(critic_embedding)['pred']

        if self._action_space == 'discrete':
            logit = self.actor_head(actor_embedding)['logit']
            return {'logit': logit.squeeze(), 'value': value}
        elif self._action_space == 'continuous':
            x = self.actor_head(actor_embedding)
            return {'logit': {'mu': x['mu'].squeeze(), 'sigma': x['sigma'].squeeze()}, 'value': value}

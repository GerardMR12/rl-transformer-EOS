import torch
import torch.nn as nn
import torch.nn.init as init

class FloatEmbedder(nn.Module):
    """"
    Class to embed the float (continuous) values of the states and actions. Child class of nn.Module.
    """
    def __init__(self, input_dim: int, embed_dim: int, num_layers: int=3, hidden_dim: int=64): # input_dim is state_dim or action_dim
        super(FloatEmbedder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2): # -2 because we already have 2 layers (input to 1st hidden and -1th hidden to output)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.embed = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for layer in self.embed:
            if isinstance(layer, nn.Linear):
                init.uniform_(layer.weight, -initrange, initrange)
                init.zeros_(layer.bias)

    def forward(self, x):
        x = self.embed(x)
        return x # ¿* math.sqrt(self.d_model)?
    
class PositionalEncoder(nn.Module):
    """
    Class to encode the position of the states and actions using the Attention Is All You Need functions. Child class of nn.Module.
    """
    def __init__(self, max_len: int, d_model: int, dropout: float):
        super(PositionalEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.empty((0, d_model), dtype=torch.float32, requires_grad=True)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0, requires_grad=True)) / d_model))

        for pos in range(max_len):
            sines = torch.sin(pos * div_term)
            cosines = torch.cos(pos * div_term)

            interleaved = torch.stack((sines, cosines), dim=1).flatten()

            pe = torch.cat((pe, interleaved.unsqueeze(0)), dim=0)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe) # register_buffer() is used to make the tensor

    def forward(self, x):
        x = (x + self.pe[:x.size(0), :])
        x = self.dropout(x)
        return x
    
class EOSTransformer(nn.Transformer):
    """
    Class to create a transformer for the Earth Observation Satellite model. Child class of nn.Transformer.
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            activation: str,
            batch_first: bool = True
        ):
        super(EOSTransformer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        self.architecture_type = "Transformer"

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Encoder
        for layer in self.encoder.layers:
            for module in layer.children():
                if type(module) == torch.nn.Linear:
                    nn.init.zeros_(module.bias)
                    nn.init.uniform_(module.weight, -initrange, initrange)

        # Decoder
        for layer in self.decoder.layers:
            for module in layer.children():
                if type(module) == torch.nn.Linear:
                    nn.init.zeros_(module.bias)
                    nn.init.uniform_(module.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
class StochasticProjector(nn.Module):
    """
    Class to project the output of the transformer into a stochastic action (twice the size of actions' size). Child class of nn.Module.
    """
    def __init__(self, d_model: int, action_dim: int):
        super(StochasticProjector, self).__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.fc = nn.Linear(d_model, int(2 * action_dim))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)

    def forward(self, x):
        x = self.fc(x).clone()
        return x
    
class EOSModel(nn.Module):
    """
    Class to create the Earth Observation Satellite model. Child class of nn.Module. Parts:
        · 1a. embedder for the states
        · 1b. embedder for the actions
        · 2. positional encoder
        · 3. transformer
        · 4. stochastic projector
    """
    def __init__(
            self,
            state_embedder: FloatEmbedder,
            action_embedder: FloatEmbedder,
            pos_encoder: PositionalEncoder,
            transformer: EOSTransformer,
            projector: StochasticProjector
        ):
        super(EOSModel, self).__init__()
        self.model_type = "Earth Observation Model"
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        self.pos_encoder = pos_encoder
        self.transformer = transformer
        self.projector = projector
        self.epsilon = torch.distributions.Normal(0, 1)

        # Check it is a transformer
        if self.transformer.architecture_type != "Transformer":
            raise ValueError("The model is not a transformer")

    def forward(self, states, actions):
        with torch.no_grad():
            # Check if states and actions have the same number of dimensions
            if states.dim() != actions.dim():
                raise ValueError("For cleaner code, the number of dimensions of states and actions should be the same")
            
            # Check whether they are batched or not
            if states.dim() == 1:
                states = states.unsqueeze(0).unsqueeze(0)
                actions = actions.unsqueeze(0).unsqueeze(0)
            elif states.dim() == 2:
                states = states.unsqueeze(0)
                actions = actions.unsqueeze(0)
            
            # Check if the number of states and actions are equal
            if states.shape[1] != actions.shape[1]:
                raise ValueError("The number of states and actions must be equal")
            else:
                seq_len = states.shape[1]

        # Pass the states tensor through the embedding layer
        embedded_states = self.state_embedder(states)

        # Pass the actions tensor through the embedding layer
        embedded_actions = self.action_embedder(actions)

        # Pass the embedded states and actions through the positional encoder
        input_states = self.pos_encoder(embedded_states)
        input_actions = self.pos_encoder(embedded_actions)

        # Set the source and target masks
        mask = self.transformer._generate_square_subsequent_mask(seq_len)

        # Pass the input states and actions through the transformer
        output = self.transformer(input_states, input_actions, src_mask=mask, tgt_mask=mask, src_is_causal=True, tgt_is_causal=True)

        # Pass the output through the projector
        stochastic_actions = self.projector(output)

        # Group the actions by features with their mean and variance
        stochastic_actions = stochastic_actions.view(-1, seq_len, actions.shape[-1], 2)

        return stochastic_actions
        
    def sample(self, stochastic_actions, batched):
        if batched:
            # Create a sampled actions tensor with the same shape as the stochastic actions tensor
            _, seq_len, action_dim, _ = stochastic_actions.shape
            sampled_actions = torch.empty((0, seq_len, action_dim), requires_grad=True)

            for batch in stochastic_actions: # for each batch (set of sequences)
                # Create a tensor for the batch
                batch_sampled_actions = torch.empty((0, action_dim), requires_grad=True)
                for action in batch:
                    # Create a tensor for the sequence
                    seq_sampled_actions = torch.empty((0), requires_grad=True)
                    for feature in action:
                        mean = feature[0]
                        log_std = feature[1]
                        std = torch.exp(log_std)

                        result = mean + std * self.epsilon.rsample()
                        seq_sampled_actions = torch.cat([seq_sampled_actions, result.unsqueeze(0)], dim=0)
                    
                    batch_sampled_actions = torch.cat([batch_sampled_actions, seq_sampled_actions.unsqueeze(0)], dim=0)
                
                sampled_actions = torch.cat([sampled_actions, batch_sampled_actions.unsqueeze(0)], dim=0)

            return sampled_actions
        else:
            # Create a sampled actions tensor with the same shape as the stochastic actions tensor
            _, action_dim, _ = stochastic_actions.shape
            sampled_actions = torch.empty((0, action_dim), requires_grad=True)

            for action in stochastic_actions:
                # Create a tensor for the action
                action_sampled_actions = torch.empty((0), requires_grad=True)
                for feature in action:
                    mean = feature[0]
                    log_std = feature[1]
                    std = torch.exp(log_std)

                    result = mean + std * self.epsilon.rsample()
                    action_sampled_actions = torch.cat([action_sampled_actions, result.unsqueeze(0)], dim=0)

                sampled_actions = torch.cat([sampled_actions, action_sampled_actions.unsqueeze(0)], dim=0)

            return sampled_actions
    
    def reparametrization_trick(self, stochastic_actions):
        # Number of dimensions of the tensor
        dim = stochastic_actions.dim()

        if dim == 4:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions, True) # already in a sequence and batched
        elif dim == 3:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions, False)
        elif dim == 2:
            # Create the specific output actions tensor
            sampled_actions = self.sample(stochastic_actions.unsqueeze(0), False).squeeze() # not in a sequence nor batched
        else:
            raise ValueError("The tensor must have 2, 3 or 4 dimensions")
        
        return sampled_actions, torch.tanh(sampled_actions)
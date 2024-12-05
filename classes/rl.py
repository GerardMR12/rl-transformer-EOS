import torch
import torchviz
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from classes.client import Client
from classes.model import *
from classes.utils import *

RT = 6371.0 # Earth radius in km

class Actor(nn.Module):
    """
    Class to represent an Actor (policy, model) in the context of the SAC algorithm. Children class of nn.Module.
    """
    def __init__(self, model: EOSModel, lr: float=1e-3):
        super(Actor, self).__init__()
        self.role_type = "Actor"
        self.model = model
        self.lr = lr

    def forward(self, states, actions):
        return self.model(states, actions)

class RL():
    """
    Class to represent the Soft Actor-Critic algorithm. Children class of nn.Module.
    """
    def __init__(self, conf: DataFromJSON, client: Client, save_path: str):
        self.__role_type = "Reinforcement Learning"
        self.__conf = conf
        self.client = client
        self.save_path = save_path
        self.set_properties(conf)
        self.losses = {"pi": []}
        self.tensor_manager = TensorManager()

    def __str__(self) -> str:
        return f"{self.__role_type} object with configuration: {self.__conf}"

    def set_properties(self, conf: DataFromJSON):
        """
        Set the properties of the SAC object.
        """
        for key, value in conf.__dict__.items():
            if not key.startswith("__"):
                setattr(self, key, value)

    def start(self):
        """
        Start the SAC algorithm.
        """
        # Create the agent and the critics
        actor = self.create_entities()

        # Warm up the agent
        list_states, list_actions = self.warm_up(actor)

        # Train the agent
        actor = self.train(actor, list_states, list_actions)

        # Plot the losses
        self.plot_losses(self.losses)

        # Save the model
        self.save_model(actor)

    def create_entities(self) -> Actor:
        """
        Create the entities for the SAC algorithm.
        """
        # Create the embedder object for states
        states_embedder = FloatEmbedder(
            input_dim=self.state_dim,
            embed_dim=self.d_model
        )
        
        # Create the embedder object for actions
        actions_embedder = FloatEmbedder(
            input_dim=self.action_dim,
            embed_dim=self.d_model
        )
        
        # Create the positional encoder object
        pos_encoder = PositionalEncoder(
            d_model=self.d_model,
            max_len=self.max_len,
            dropout=self.pos_dropout
        )

        # Create the transformer model
        transformer = EOSTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.activation,
            batch_first=self.batch_first
        )
        
        # Create a linear outside stochastic layer called projector
        stochastic_projector = StochasticProjector(
            d_model=self.d_model,
            action_dim=self.action_dim
        )
        
        # Create the model object
        model = EOSModel(
            state_embedder=states_embedder,
            action_embedder=actions_embedder,
            pos_encoder=pos_encoder,
            transformer=transformer,
            projector=stochastic_projector
        )

        # Create the actor
        actor = Actor(model, lr=self.lambda_pi)
        
        # Load the previous models if they exist
        if os.path.exists(self.save_path) and self.load_model and os.path.exists(f"{self.save_path}\\model.pth"):
            print("Loading previous models...")
            actor.model.load_state_dict(torch.load(f"{self.save_path}\\model.pth", weights_only=True))

        return actor

    def warm_up(self, actor: Actor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Warm up the SAC algorithm before starting the training loop.
        """
        torch.autograd.set_detect_anomaly(True)

        list_states: list[torch.Tensor] = []
        list_actions: list[torch.Tensor] = []

        # Loop over all agents
        for agt in self.agents:
            # Sending data to get the initial state
            sending_data = {
                "agent_id": agt,
                "action": {
                    "d_pitch": 0,
                    "d_roll": 0
                },
                "delta_time": 0
            }
            state, _, _ = self.client.get_next_state("get_next", sending_data)

            # Normalize the state given by the environment
            vec_state = self.normalize_state(state)

            # Input tensor of 1 batch and 1 sequence of state_dim dimensional states
            states = torch.FloatTensor([[vec_state]])

            # Input tensor of 1 batch and 1 sequence of action_dim dimensional actions (equal to 0)
            actions = torch.FloatTensor([[[0 for _ in range(self.action_dim)]]])

            list_states += [states]
            list_actions += [actions]

        # Loop flags
        done = False

        print("Starting warm-up...")

        # Loop over all iterations
        for _ in range(self.warm_up_steps):
            # Loop over all agents
            for idx, agt in enumerate(self.agents):
                states = list_states[idx]
                actions = list_actions[idx]

                with torch.no_grad():
                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :]
                    actions = actions[:, -self.max_len:, :]

                    # Create the augmented state
                    aug_state = [states.clone(), actions.clone()]

                    # Get the stochastic actions
                    stochastic_actions = actor(states, actions)

                    # Select the last stochastic action
                    a_sto = stochastic_actions[-1, -1, :]

                    # Sample and convert the action
                    _, a = actor.model.reparametrization_trick(a_sto)

                    # --------------- Environment's job to provide info ---------------
                    sending_data = {
                        "agent_id": agt,
                        "action": {
                            "d_pitch": a[0].item() * 30,
                            "d_roll": a[1].item() * 60
                        },
                        "delta_time": self.time_increment
                    }
                    
                    state, _, done = self.client.get_next_state("get_next", sending_data)

                    # Break if time is up
                    if done:
                        print("Time is up!")
                        break

                    # Normalize the state
                    vec_state = self.normalize_state(state)

                    # Get the next state
                    s_next = torch.FloatTensor(vec_state)
                    # --------------- Environment's job to provide info ---------------

                    # Add it to the states
                    states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0)], dim=1)

                    # Add it to the actions
                    actions = torch.cat([actions, a.unsqueeze(0).unsqueeze(0)], dim=1)

                    # Adjust the maximum length of the states and actions
                    states = states[:, -self.max_len:, :]
                    actions = actions[:, -self.max_len:, :]

                    # Replace the states and actions lists
                    list_states[idx] = states
                    list_actions[idx] = actions

        print("✔ Warm-up done!")
        
        return list_states, list_actions

    def train(self, actor: Actor, list_states: list[torch.Tensor], list_actions: list[torch.Tensor]):
        """
        Begin the training of the SAC algorithm.
        """
        torch.autograd.set_detect_anomaly(True)

        # Optimizers
        optimizer_pi = optim.Adam(actor.model.parameters(), lr=actor.lr)

        # Loop flags
        done = False
        iteration = 1

        print("Starting training...")

        # Loop over all iterations
        while not done:
            print(f"\nStarting iteration {iteration}...")
            iteration += 1

            for idx, agt in enumerate(self.agents):
                states = list_states[idx]
                actions = list_actions[idx]

                # Adjust the maximum length of the states and actions
                states = states[:, -self.max_len:, :]
                actions = actions[:, -self.max_len:, :]

                # Get the stochastic actions
                stochastic_actions = actor(states, actions)

                # Select the last stochastic action
                a_sto = stochastic_actions[-1, -1, :]

                # Sample and convert the action
                _, a = actor.model.reparametrization_trick(a_sto)

                # --------------- Environment's job to provide info ---------------
                sending_data = {
                    "agent_id": agt,
                    "action": {
                        "d_pitch": a[0].item() * 30,
                        "d_roll": a[1].item() * 60
                    },
                    "delta_time": self.time_increment
                }
                
                state, reward, done = self.client.get_next_state("get_next", sending_data)

                # Break if time is up
                if done:
                    print("Time is up!")
                    break

                # Normalize the state
                vec_state = self.normalize_state(state)

                # Get the reward
                r = torch.FloatTensor([reward * self.reward_scale])

                # Get the next state
                s_next = torch.FloatTensor(vec_state)
                # --------------- Environment's job to provide info ---------------

                # Add it to the states
                states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0)], dim=1)

                # Add it to the actions
                actions = torch.cat([actions, a.unsqueeze(0).unsqueeze(0)], dim=1)

                # Adjust the maximum length of the states and actions
                states = states[:, -self.max_len:, :]
                actions = actions[:, -self.max_len:, :]

                # Replace the states and actions lists
                list_states[idx] = states
                list_actions[idx] = actions

                # Set the gradients to zero
                optimizer_pi.zero_grad()

                # Compute the losses
                J_pi = r

                # Store the losses
                self.losses["pi"].append(J_pi.item())

                # Backpropagate
                J_pi.backward(retain_graph=True)

                # Optimize parameters
                optimizer_pi.step()

                print("✔ Iteration done!")

                # Break if time is up
                if done:
                    break

        return actor
    
    def normalize_state(self, state: dict) -> list:
        """
        Normalize the action dictionary to a list.
        """
        # Conversion dictionary: each has two elements, the first is the gain and the second is the offset
        conversion_dict = {
            "a": (1/RT, -1), "e": (1, 0), "i": (1/180, 0), "raan": (1/360, 0), "aop": (1/360, 0), "ta": (1/360, 0), # orbital elements
            "az": (1/360, 0), "el": (1/180, 0.5), # azimuth and elevation
            "pitch": (1/180, 0.5), "roll": (1/360, 0.5), # attitude
            "detic_lat": (1/180, 0.5), "detic_lon": (1/360, 0), "detic_alt": (1/RT, 0), # nadir position
            "lat": (1/180, 0.5), "lon": (1/360, 0), "priority": (1/10, 0) # targets clues
        }

        vec_state = []
        for key, value in state.items():
            if key.startswith("lat_") or key.startswith("lon_") or key.startswith("priority_"):
                key = key.split("_")[0]
            vec_state.append(value * conversion_dict[key][0] + conversion_dict[key][1])

        return vec_state
    
    def plot_losses(self, losses: dict):
        """
        Plot the losses.
        """
        smoothed_pi = pd.DataFrame(losses["pi"]).rolling(window=int(len(losses["pi"])/10)).mean()

        plt.plot(smoothed_pi)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Rewards over time")

        plt.savefig(f"{self.save_path}\\losses.png", dpi=500)
    
    def save_model(self, actor: Actor):
        """
        Save the model to the specified path.
        """
        torch.save(actor.model.state_dict(), f"{self.save_path}\\model.pth")
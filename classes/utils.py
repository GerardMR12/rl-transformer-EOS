import torch

class TensorManager():
    """
    Class to manage the tensors with overall applications.
    """
    def __init__(self):
        self.object_type = "Tensor Manager"

    def full_squeeze(self, *tensors):
        """
        Squeeze tensor until it does not have any dimension of size 1 or squeeze the tensors in tuples until they have 3 dimensions (transformer requirements).
        """
        elements = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                while tensor.dim() != 0 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    while tensor[i].dim() != 0 and tensor[i].shape[0] == 1:
                        tensor[i] = tensor[i].squeeze(0)

            elements.append(tensor)

        return elements
    
    def batchify(self, *tensors):
        """
        Concatenate tensors into a single tensor.
        """
        elements = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                while tensor.dim() < 3:
                    tensor = tensor.unsqueeze(0)
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    while tensor[i].dim() < 3:
                        tensor[i] = tensor[i].unsqueeze(0)

            elements.append(tensor)

        return elements
    
class DataFromJSON():
    """
    Class to manage the data of the model.
    """
    def __init__(self, json_dict, data_type: str):
        self.__class_name = "DataFromJSON"
        self.__data_type = data_type
        self.loop(json_dict)

    def __str__(self):
        return f"{self.__class_name} object with data type: {self.__data_type}"

    def loop(self, json_dict):
        if not isinstance(json_dict, dict):
            return
        for key, value in json_dict.items():
            if isinstance(value, dict):
                self.loop(value)
            else:
                if hasattr(self, key):
                    raise ValueError(f"Variable {key} already exists in the class. Rename the json key in your configuration file.")
                else:
                    setattr(self, key, value)